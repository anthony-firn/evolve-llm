//! `pi TASK`: give a task to Pi, the coding agent, from the Evolve VM's shell. Pi runs on the page
//! with the loaded model, works in this directory with the VM's shell and files (and git and gh),
//! and writes its progress here. Exits 0 when Pi finished, 1 when it stopped on an error.

use std::env;
use std::io::{self, Read};

#[link(wasm_import_module = "evolve")]
extern "C" {
    fn agent_run(task: *const u8, task_len: usize, cwd: *const u8, cwd_len: usize, status: *mut i32) -> u32;
}

fn main() {
    if let Ok(p) = env::var("PWD") {
        let _ = env::set_current_dir(p);
    }
    let mut task: String = env::args().skip(1).collect::<Vec<_>>().join(" ");
    if task == "-" {
        task.clear();
        io::stdin().read_to_string(&mut task).ok();
    }
    if task.trim().is_empty() || task == "--help" || task == "-h" {
        println!("usage: pi TASK    (or: pi - < task.txt)\nPi works in the current directory with the shell, files, git and gh, using the model loaded on the page.");
        std::process::exit(if task.trim().is_empty() { 2 } else { 0 });
    }
    let cwd = env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| "/home/user".into());
    let mut status = 1i32;
    let e = unsafe { agent_run(task.as_ptr(), task.len(), cwd.as_ptr(), cwd.len(), &mut status) };
    if e != 0 {
        eprintln!("pi: could not reach the agent (error {e})");
        std::process::exit(1);
    }
    std::process::exit(status);
}
