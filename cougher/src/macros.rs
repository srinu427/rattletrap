#[macro_export]
macro_rules! build_init_cleanup_struct {
    ($name: tt, $ty: ty, $sel:ident, $drop_fn: stmt) => {
        pub struct $name<'a> {
            pub drop: bool,
            pub inner: $ty,
            pub device: &'a ash::Device,
        }

        impl<'a> $name<'a> {
            pub fn take(mut $sel) -> $ty {
                $sel.drop = false;
                $sel.inner
            }

            pub fn read(& $sel) -> $ty {
                $sel.inner
            }
        }

        impl<'a> Drop for $name<'a> {
            fn drop(&mut $sel) {
                if $sel.drop {
                    unsafe {
                        $drop_fn
                    }
                }
            }
        }
    };
}
