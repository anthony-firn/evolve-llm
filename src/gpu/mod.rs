//! WebGPU device access through wgpu. The same code drives Vulkan, Metal or
//! D3D12 natively and the browser's WebGPU implementation from WebAssembly.

pub mod blocks;
pub mod engine;
pub mod shaders;

use serde::Serialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

#[derive(Clone, Debug, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub architecture: String,
    pub backend: String,
    pub driver: String,
    pub max_buffer_size: u64,
    pub max_storage_binding: u64,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: DeviceInfo,
    pub error: Arc<Mutex<Option<String>>>,
}

fn power_preference() -> wgpu::PowerPreference {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("EVOLVE_LLM_POWER").as_deref() == Ok("low") {
        return wgpu::PowerPreference::LowPower;
    }
    wgpu::PowerPreference::HighPerformance
}

pub async fn init() -> Result<Gpu, String> {
    init_with(power_preference()).await
}

/// Open a GPU with an explicit preference; laptops often have an integrated and a discrete one.
pub async fn init_with(preference: wgpu::PowerPreference) -> Result<Gpu, String> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    #[cfg(target_arch = "wasm32")]
    {
        desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY);
    }
    let instance = wgpu::Instance::new(desc);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: preference,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| format!("no WebGPU adapter is available ({e})"))?;
    let al = adapter.limits();
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
    limits.max_buffer_size = al.max_buffer_size;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("evolve-llm"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("could not open the GPU: {e}"))?;
    let error = Arc::new(Mutex::new(None));
    let sink = error.clone();
    device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        let mut slot = sink.lock().unwrap();
        if slot.is_none() {
            *slot = Some(e.to_string());
        }
    }));
    let ai = adapter.get_info();
    let dl = device.limits();
    Ok(Gpu {
        device,
        queue,
        info: DeviceInfo {
            name: ai.name,
            vendor: if ai.vendor == 0 {
                String::new()
            } else {
                format!("{:#06x}", ai.vendor)
            },
            architecture: format!("{:?}", ai.device_type),
            backend: format!("{:?}", ai.backend),
            driver: ai.driver_info,
            max_buffer_size: dl.max_buffer_size,
            max_storage_binding: dl.max_storage_buffer_binding_size,
        },
        error,
    })
}

impl Gpu {
    pub fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap().take()
    }
}

struct Shared<T> {
    value: Option<T>,
    waker: Option<Waker>,
}

/// Resolves when a wgpu callback fires; works with and without threads.
pub struct Oneshot<T>(Arc<Mutex<Shared<T>>>);

pub fn oneshot<T>() -> (impl FnOnce(T) + Send + 'static, Oneshot<T>)
where
    T: Send + 'static,
{
    let shared = Arc::new(Mutex::new(Shared {
        value: None,
        waker: None,
    }));
    let tx = shared.clone();
    let send = move |v: T| {
        let mut s = tx.lock().unwrap();
        s.value = Some(v);
        if let Some(w) = s.waker.take() {
            w.wake();
        }
    };
    (send, Oneshot(shared))
}

impl<T> Future for Oneshot<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut s = self.0.lock().unwrap();
        match s.value.take() {
            Some(v) => Poll::Ready(v),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Map a MAP_READ buffer and copy its first `len` bytes out.
pub async fn read_back(gpu: &Gpu, buffer: &wgpu::Buffer, len: u64) -> Result<Vec<u8>, String> {
    let (send, done) = oneshot();
    buffer
        .slice(0..len)
        .map_async(wgpu::MapMode::Read, move |r| {
            send(r.map_err(|e| e.to_string()))
        });
    #[cfg(not(target_arch = "wasm32"))]
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    done.await?;
    let data = buffer
        .slice(0..len)
        .get_mapped_range()
        .map_err(|e| e.to_string())?
        .to_vec();
    buffer.unmap();
    if let Some(e) = gpu.take_error() {
        return Err(format!("GPU error: {e}"));
    }
    Ok(data)
}
