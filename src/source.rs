//! Where weights are read from while a model runs: weights that do not fit in GPU memory are
//! read again on demand, so the operating system's file cache acts as a RAM tier in front of
//! the disk.

/// Random-access reads from the files of a model (one file, or the parts of a split model).
pub trait WeightSource {
    fn read(&self, file: usize, offset: u64, len: usize) -> Result<Vec<u8>, String>;
}

/// Whole files held in memory (tests and small models).
pub struct MemorySource(pub Vec<Vec<u8>>);

impl WeightSource for MemorySource {
    fn read(&self, file: usize, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let f = self.0.get(file).ok_or("no such file")?;
        let start = offset as usize;
        f.get(start..start + len)
            .map(|b| b.to_vec())
            .ok_or_else(|| "read past the end of the model file".into())
    }
}

/// Files on disk, read with positioned reads.
#[cfg(not(target_arch = "wasm32"))]
pub struct FileSource {
    files: Vec<std::fs::File>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileSource {
    pub fn open(paths: &[std::path::PathBuf]) -> Result<FileSource, String> {
        let files = paths
            .iter()
            .map(|p| std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display())))
            .collect::<Result<_, _>>()?;
        Ok(FileSource { files })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl WeightSource for FileSource {
    fn read(&self, file: usize, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let f = self.files.get(file).ok_or("no such file")?;
        let mut buf = vec![0u8; len];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            f.read_exact_at(&mut buf, offset).map_err(|e| e.to_string())?;
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut done = 0;
            while done < len {
                let n = f
                    .seek_read(&mut buf[done..], offset + done as u64)
                    .map_err(|e| e.to_string())?;
                if n == 0 {
                    return Err("read past the end of the model file".into());
                }
                done += n;
            }
        }
        Ok(buf)
    }
}
