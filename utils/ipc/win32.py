"""纯 ctypes 的 kernel32 绑定（零第三方依赖）。

所有 API 显式声明 argtypes/restype——不设则默认 c_int 返回，
64 位 HANDLE 高位会被截断（MapViewOfFile 返回的地址尤其如此）。
错误处理：返回 NULL/0/False 时抛 WinAPIError（基于 use_last_error）。
"""
import ctypes
import ctypes.wintypes as wt

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# ---- 常量（协议与 win32 API） ----
PAGE_READWRITE = 0x04
FILE_MAP_ALL_ACCESS = 0xF001F
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1)
ERROR_ALREADY_EXISTS = 183
INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102
WAIT_ABANDONED = 0x80
WAIT_FAILED = 0xFFFFFFFF
EVENT_ALL_ACCESS = 0x1F0003
MUTEX_ALL_ACCESS = 0x1F0001

_LPCWSTR = ctypes.c_wchar_p
_LPVOID = ctypes.c_void_p
_LPCVOID = ctypes.c_void_p


class WinAPIError(OSError):
    """Win32 调用失败，errno 为 GetLastError()。"""


def _check_handle(result, func_name):
    if not result:
        raise WinAPIError(func_name, ctypes.get_last_error())
    return result


def _check_bool(result, func_name):
    if not result:
        raise WinAPIError(func_name, ctypes.get_last_error())
    return True


# ---- 内存映射 ----
kernel32.CreateFileMappingW.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD,
                                        wt.DWORD, wt.DWORD, _LPCWSTR]
kernel32.CreateFileMappingW.restype = wt.HANDLE


def create_file_mapping(name: str, size: int) -> tuple:
    """创建命名映射（页文件 backed）。返回 (handle, created)。

    created=False 表示对象已存在（ERROR_ALREADY_EXISTS）。
    返回的 handle 归调用者所有（须 CloseHandle），不得丢弃——
    丢弃会使命名对象在本进程内永不销毁。
    注意：get_last_error 必须在调用后立即读取，中间不得插其他 ctypes 调用。
    """
    handle = kernel32.CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE,
                                         size >> 32, size & 0xFFFFFFFF, name)
    existed = ctypes.get_last_error() == ERROR_ALREADY_EXISTS
    _check_handle(handle, "CreateFileMappingW")
    return handle, not existed


kernel32.OpenFileMappingW.argtypes = [wt.DWORD, wt.BOOL, _LPCWSTR]
kernel32.OpenFileMappingW.restype = wt.HANDLE


def open_file_mapping(name: str) -> int:
    handle = kernel32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, name)
    return _check_handle(handle, "OpenFileMappingW")


kernel32.MapViewOfFile.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD,
                                   ctypes.c_size_t]
kernel32.MapViewOfFile.restype = _LPVOID  # 必须 LPVOID，否则 64 位地址被 c_int 截断


def map_viewOfFile(handle: int, size: int = 0) -> int:
    """映射整个视图（size=0）。返回基地址（int）。"""
    base = kernel32.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size)
    return _check_handle(base, "MapViewOfFile")


kernel32.UnmapViewOfFile.argtypes = [_LPCVOID]
kernel32.UnmapViewOfFile.restype = wt.BOOL


def unmap_view(base: int) -> None:
    kernel32.UnmapViewOfFile(_LPCVOID(base))


kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.CloseHandle.restype = wt.BOOL


def close_handle(handle: int) -> None:
    if handle:
        kernel32.CloseHandle(handle)


# ---- 事件 / 互斥体 ----
kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.BOOL, _LPCWSTR]
kernel32.CreateEventW.restype = wt.HANDLE


def create_event(name: str) -> int:
    """manual-reset、初始无信号。已存在时同名打开（属性不匹配由调用方保证一致）。"""
    handle = kernel32.CreateEventW(None, True, False, name)
    return _check_handle(handle, "CreateEventW")


kernel32.OpenEventW.argtypes = [wt.DWORD, wt.BOOL, _LPCWSTR]
kernel32.OpenEventW.restype = wt.HANDLE


def open_event(name: str) -> int:
    handle = kernel32.OpenEventW(EVENT_ALL_ACCESS, False, name)
    return _check_handle(handle, "OpenEventW")


kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wt.BOOL, _LPCWSTR]
kernel32.CreateMutexW.restype = wt.HANDLE


def create_mutex(name: str) -> int:
    handle = kernel32.CreateMutexW(None, False, name)
    return _check_handle(handle, "CreateMutexW")


kernel32.OpenMutexW.argtypes = [wt.DWORD, wt.BOOL, _LPCWSTR]
kernel32.OpenMutexW.restype = wt.HANDLE


def open_mutex(name: str) -> int:
    handle = kernel32.OpenMutexW(MUTEX_ALL_ACCESS, False, name)
    return _check_handle(handle, "OpenMutexW")


kernel32.SetEvent.argtypes = [wt.HANDLE]
kernel32.SetEvent.restype = wt.BOOL
kernel32.ResetEvent.argtypes = [wt.HANDLE]
kernel32.ResetEvent.restype = wt.BOOL
kernel32.ReleaseMutex.argtypes = [wt.HANDLE]
kernel32.ReleaseMutex.restype = wt.BOOL


def set_event(handle: int) -> None:
    _check_bool(kernel32.SetEvent(handle), "SetEvent")


def reset_event(handle: int) -> None:
    _check_bool(kernel32.ResetEvent(handle), "ResetEvent")


def release_mutex(handle: int) -> None:
    _check_bool(kernel32.ReleaseMutex(handle), "ReleaseMutex")


kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
kernel32.WaitForSingleObject.restype = wt.DWORD


def wait_single(handle: int, timeout_ms: int) -> int:
    """ctypes 外呼释放 GIL；返回 WAIT_OBJECT_0 / WAIT_TIMEOUT / WAIT_ABANDONED / WAIT_FAILED。"""
    return kernel32.WaitForSingleObject(handle, timeout_ms)


# ---- 时钟 ----
kernel32.GetTickCount64.argtypes = []
kernel32.GetTickCount64.restype = ctypes.c_uint64


def GetTickCount64() -> int:  # noqa: N802 - 保留 WinAPI 原名
    """系统启动以来的毫秒数（跨进程一致的心跳时钟）。"""
    return kernel32.GetTickCount64()
