//! Transfer counters for each layer of the Snowflake stack (Evolve Browser diagnostics).

use std::cell::Cell;

#[derive(Default)]
pub struct Stats {
    pub ws_rx: Cell<u64>,
    pub ws_tx: Cell<u64>,
    pub ws_rx_msgs: Cell<u64>,
    pub kcp_input: Cell<u64>,
    pub kcp_recv: Cell<u64>,
    pub kcp_send: Cell<u64>,
    pub kcp_wait_snd: Cell<u64>,
    pub kcp_rmt_wnd: Cell<u64>,
    pub kcp_flushes: Cell<u64>,
    pub smux_psh_in: Cell<u64>,
    pub smux_psh_out: Cell<u64>,
    pub smux_upd_sent: Cell<u64>,
    pub smux_upd_consumed: Cell<u64>,
    pub smux_peer_window: Cell<u64>,
    pub smux_peer_consumed: Cell<u64>,
    pub smux_frames_in: Cell<u64>,
}

thread_local! {
    pub static STATS: Stats = Stats::default();
}

pub fn add(f: impl FnOnce(&Stats) -> &Cell<u64>, n: u64) {
    STATS.with(|s| {
        let c = f(s);
        c.set(c.get() + n)
    });
}

pub fn set(f: impl FnOnce(&Stats) -> &Cell<u64>, n: u64) {
    STATS.with(|s| f(s).set(n));
}

pub fn json() -> String {
    STATS.with(|s| {
        format!(
            "{{\"ws_rx\":{},\"ws_tx\":{},\"ws_rx_msgs\":{},\"kcp_input\":{},\"kcp_recv\":{},\"kcp_send\":{},\"kcp_wait_snd\":{},\"kcp_rmt_wnd\":{},\"kcp_flushes\":{},\"smux_psh_in\":{},\"smux_psh_out\":{},\"smux_upd_sent\":{},\"smux_upd_consumed\":{},\"smux_peer_window\":{},\"smux_peer_consumed\":{},\"smux_frames_in\":{}}}",
            s.ws_rx.get(), s.ws_tx.get(), s.ws_rx_msgs.get(), s.kcp_input.get(), s.kcp_recv.get(), s.kcp_send.get(),
            s.kcp_wait_snd.get(), s.kcp_rmt_wnd.get(), s.kcp_flushes.get(), s.smux_psh_in.get(), s.smux_psh_out.get(),
            s.smux_upd_sent.get(), s.smux_upd_consumed.get(), s.smux_peer_window.get(), s.smux_peer_consumed.get(), s.smux_frames_in.get()
        )
    })
}
