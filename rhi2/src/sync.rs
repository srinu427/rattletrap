#[derive(Debug, thiserror::Error)]
pub enum SyncErr {
    #[error("waiting for cpu-future failed: {0}")]
    WaitErr(String),
}

pub trait CpuFuture {
    fn wait(&mut self);
}

pub trait GpuFuture {}

pub enum PipelineStage {
    Top,
    Vertex,
    DepthTest,
    Fragment,
    AttachWrite,
    Transfer,
    End,
}
