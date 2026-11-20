fn main() {
    #[cfg(feature = "s3")]
    {
        let headers = std::env::var_os("DEP_HDF5_INCLUDE").expect("bundled HDF5 headers");
        println!("cargo:rerun-if-changed=src/range_vfd.c");
        cc::Build::new()
            .include(headers)
            .file("src/range_vfd.c")
            .compile("h5rs_range_vfd");
    }
}
