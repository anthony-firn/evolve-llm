// Saved conversations, kept only in this browser (IndexedDB), so the person can come back to them:
// {id, title, updated, model, messages: [{role, content, stats?, steps?, preview?}]}.

const DB = 'evolve-llm-chats';
const STORE = 'chats';

function open() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'id' });
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error || new Error('Browser storage is unavailable.'));
  });
}

async function run(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const request = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); resolve(request.result); };
    t.onerror = () => { db.close(); reject(t.error); };
  });
}

export const newChatId = () => crypto.randomUUID();
export const saveChat = (chat) => run('readwrite', (s) => s.put(chat));
export const getChat = (id) => run('readonly', (s) => s.get(id));
export const deleteChat = (id) => run('readwrite', (s) => s.delete(id));
/** Newest first. */
export const listChats = async () => (await run('readonly', (s) => s.getAll())).sort((a, b) => b.updated - a.updated);
