#[derive(Debug, thiserror::Error)]
pub enum SyncErr {
    #[error("waiting for cpu-future failed: {0}")]
    WaitErr(String),
}

pub trait TaskFuture {
    fn wait(&mut self) -> Result<(), SyncErr>;
}

pub enum PipelineStage {
    Top,
    Vertex,
    DepthTest,
    Fragment,
    AttachWrite,
    Transfer,
    Bottom,
}
