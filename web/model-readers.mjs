// Model files need random reads, never write permission. Keep the direct-memory fast path
// where supported, and use bounded FileReaderSync slices when a browser cannot share a lock.
export function fileReader(file, SyncReader = globalThis.FileReaderSync) {
  const sync = new SyncReader();
  return {
    size: file.size,
    read(offset, target) {
      const buf = sync.readAsArrayBuffer(file.slice(offset, offset + target.length));
      if (buf.byteLength !== target.length) throw new Error('The file is shorter than its header says. It may be incomplete.');
      target.set(new Uint8Array(buf));
    },
    close() {},
  };
}

export async function storedReader(handle, SyncReader = globalThis.FileReaderSync) {
  let access;
  try {
    access = await handle.createSyncAccessHandle({ mode: 'read-only' });
  } catch (error) {
    // In particular, Firefox may reject a second handle even with read-only requested.
    // Do not retry with the default readwrite mode or ask for unsafe write access.
    if (!['NoModificationAllowedError', 'NotSupportedError', 'TypeError'].includes(error?.name)) throw error;
    return fileReader(await handle.getFile(), SyncReader);
  }
  try {
    return {
      size: access.getSize(),
      read(offset, target) {
        if (access.read(target, { at: offset }) !== target.length) {
          throw new Error('The stored model is shorter than its header says.');
        }
      },
      close() { access.close(); },
    };
  } catch (error) {
    access.close();
    throw error;
  }
}

export async function openModelReaders(source, modelsDirectory, SyncReader = globalThis.FileReaderSync) {
  const readers = [];
  try {
    if (source.names) {
      const directory = await modelsDirectory();
      for (const name of source.names) {
        readers.push(await storedReader(await directory.getFileHandle(name), SyncReader));
      }
    } else {
      for (const file of source.files) readers.push(fileReader(file, SyncReader));
    }
    return readers;
  } catch (error) {
    for (const reader of readers) { try { reader.close(); } catch {} }
    throw error;
  }
}

export function storageError(error) {
  return error?.name === 'NoModificationAllowedError'
    ? 'Browser storage is locked by another tab or download. Close other Evolve LLM tabs and try again, or restart the browser. Your saved download has not been removed.'
    : String(error?.message || error);
}
