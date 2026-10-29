"""a12_retire: MegaMoE scale factors in a compressible device allocation; no import-time device work.

Storage asks the driver for a generic-compression allocation (cuMemCreate) and refuses unless the grant matches.
NativeScales copies one MN-major int32 scale image into it and checks byte equality with the source before
returning. The owner must outlive every CUDA graph that reads the tensor; there is no implicit destructor.
"""
from cuda.bindings import driver as cu

def checked(result):
    error,*values=result
    if int(error):raise RuntimeError((str(error),values))
    return values[0] if len(values)==1 else values

class Storage:
    def __init__(self,nbytes,compressed,device=0):
        self.handle=None;self.pointer=None;self.mapped=False;self.closed=False
        self.nbytes=nbytes;self.device=device
        prop=cu.CUmemAllocationProp();prop.type=cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type=cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=device
        prop.allocFlags.compressionType=int(cu.CUmemAllocationCompType.CU_MEM_ALLOCATION_COMP_GENERIC) if compressed else 0
        gran=checked(cu.cuMemGetAllocationGranularity(prop,cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        self.size=(nbytes+gran-1)//gran*gran
        try:
            self.handle=checked(cu.cuMemCreate(self.size,prop,0))
            actual=checked(cu.cuMemGetAllocationPropertiesFromHandle(self.handle))
            self.granted=int(actual.allocFlags.compressionType)
            if self.granted!=int(bool(compressed)):raise RuntimeError(('requested compression not granted',compressed,self.granted))
            self.pointer=checked(cu.cuMemAddressReserve(self.size,gran,0,0))
            checked(cu.cuMemMap(self.pointer,self.size,0,self.handle,0));self.mapped=True
            access=cu.CUmemAccessDesc();access.location=prop.location;access.flags=cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            checked(cu.cuMemSetAccess(self.pointer,self.size,[access],1))
        except BaseException:self.close();raise
    @property
    def __cuda_array_interface__(self):
        if self.closed:raise RuntimeError('compressed SF storage already closed')
        return dict(shape=(self.nbytes,),strides=None,typestr='|u1',data=(int(self.pointer),False),version=3)
    def close(self):
        if self.closed and self.pointer is None and self.handle is None:return
        self.closed=True
        failures=[]
        if self.mapped:
            try:checked(cu.cuMemUnmap(self.pointer,self.size));self.mapped=False
            except Exception as error:failures.append(str(error))
        if self.pointer is not None and not self.mapped:
            try:checked(cu.cuMemAddressFree(self.pointer,self.size));self.pointer=None
            except Exception as error:failures.append(str(error))
        if self.handle is not None:
            try:checked(cu.cuMemRelease(self.handle));self.handle=None
            except Exception as error:failures.append(str(error))
        if failures:raise RuntimeError(('compressed SF cleanup incomplete',failures))

class NativeScales:
    """Own the unchanged MN-major int32 image and its backing allocation.

    Keep this owner until every graph using tensor is retired. close() cannot
    run while a captured graph may replay; there is no implicit destructor.
    """
    def __init__(self,source,compressed=True):
        import torch
        assert source.is_cuda and source.dtype==torch.int32 and source.ndim==3
        e,n,q=source.shape
        assert source.stride()==(n*q,1,n),'native MN-major image required'
        self.storage=Storage(source.numel()*4,compressed,source.device.index)
        try:
            raw=torch.as_tensor(self.storage,device=source.device)
            assert raw.data_ptr()==int(self.storage.pointer),'interop silently copied the allocation'
            self.raw=raw;self.tensor=raw.view(torch.int32).view(e,q,n).transpose(1,2)
            self.tensor.copy_(source)
            # Force SM stores; a copy-engine transfer alone is insufficient
            # evidence that the compressed write path was populated.
            self.raw.bitwise_xor_(0)
            torch.cuda.synchronize()
            assert self.tensor.stride()==source.stride() and torch.equal(self.tensor,source)
        except BaseException:
            try:torch.cuda.synchronize()
            finally:self.storage.close()
            raise
    def close(self):
        import torch
        torch.cuda.synchronize();self.storage.close()
