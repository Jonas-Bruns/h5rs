/*
 * Read-only HDF5 driver for metadata on a byte-range source. Use the bundled
 * C headers for H5FD_class_t so the callback ABI tracks the linked HDF5.
 */
#include "hdf5.h"
#include "H5FDdevelop.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern int h5rs_range_metadata_read(const void *object, uint64_t addr, size_t size, void *buffer);

typedef struct {
    const void *object;
    uint64_t size;
} RangeAccess;

typedef struct {
    H5FD_t base;
    RangeAccess access;
    haddr_t eoa;
} RangeFile;

static void *copy_access(const void *source) {
    RangeAccess *copy = malloc(sizeof(*copy));
    if (copy)
        memcpy(copy, source, sizeof(*copy));
    return copy;
}

static herr_t free_access(void *access) {
    free(access);
    return 0;
}

static void *get_access(H5FD_t *file) {
    return copy_access(&((RangeFile *)file)->access);
}

static H5FD_t *open_file(const char *name, unsigned flags, hid_t fapl, haddr_t maxaddr) {
    const RangeAccess *access = H5Pget_driver_info(fapl);
    RangeFile *file;
    (void)name;
    if (flags != H5F_ACC_RDONLY || !access || !access->object || access->size > maxaddr)
        return NULL;
    file = calloc(1, sizeof(*file));
    if (!file)
        return NULL;
    file->access = *access;
    return &file->base;
}

static herr_t close_file(H5FD_t *file) {
    free(file);
    return 0;
}

static haddr_t get_eoa(const H5FD_t *file, H5FD_mem_t type) {
    (void)type;
    return ((const RangeFile *)file)->eoa;
}

static herr_t set_eoa(H5FD_t *file, H5FD_mem_t type, haddr_t address) {
    (void)type;
    ((RangeFile *)file)->eoa = address;
    return 0;
}

static haddr_t get_eof(const H5FD_t *file, H5FD_mem_t type) {
    (void)type;
    return ((const RangeFile *)file)->access.size;
}

static herr_t read_range(H5FD_t *file, H5FD_mem_t type, hid_t dxpl, haddr_t address,
                         size_t size, void *buffer) {
    const RangeAccess *access = &((const RangeFile *)file)->access;
    (void)type;
    (void)dxpl;
    if (size == 0)
        return 0;
    if (address > access->size || size > access->size - address)
        return -1;
    return h5rs_range_metadata_read(access->object, address, size, buffer);
}

static herr_t deny_write(H5FD_t *file, H5FD_mem_t type, hid_t dxpl, haddr_t address,
                         size_t size, const void *buffer) {
    (void)file;
    (void)type;
    (void)dxpl;
    (void)address;
    (void)size;
    (void)buffer;
    return -1;
}

/* Called under the h5rs HDF5 lock. The Rust Object outlives the file and FAPL. */
herr_t h5rs_set_range_driver(hid_t fapl, const void *object, uint64_t size) {
    static hid_t id = H5I_INVALID_HID;
    RangeAccess access = {object, size};
    if (id < 0) {
        H5FD_class_t driver = {0};
        const H5FD_mem_t map[H5FD_MEM_NTYPES] = H5FD_FLMAP_SINGLE;
        driver.version = H5FD_CLASS_VERSION;
        driver.value = H5_VFD_MAX;
        driver.name = "h5rs-range";
        driver.maxaddr = INT64_MAX;
        driver.fc_degree = H5F_CLOSE_WEAK;
        driver.fapl_size = sizeof(RangeAccess);
        driver.fapl_get = get_access;
        driver.fapl_copy = copy_access;
        driver.fapl_free = free_access;
        driver.open = open_file;
        driver.close = close_file;
        driver.get_eoa = get_eoa;
        driver.set_eoa = set_eoa;
        driver.get_eof = get_eof;
        driver.read = read_range;
        driver.write = deny_write;
        memcpy(driver.fl_map, map, sizeof(map));
        id = H5FDregister(&driver);
        if (id < 0)
            return -1;
    }
    return H5Pset_driver(fapl, id, &access);
}
