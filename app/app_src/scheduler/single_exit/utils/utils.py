def copy_mp_list(dest, src, lock):
    with lock:
        dest[:] = []
        dest.extend(src)
