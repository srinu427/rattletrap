#[macro_export]
macro_rules! make_init_struct_copy {
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

#[macro_export]
macro_rules! make_init_struct {
    ($name: tt, $ty: ty, $sel:ident, $drop_fn: stmt) => {
        pub struct $name<'a> {
            pub drop: bool,
            pub inner: std::mem::ManuallyDrop<$ty>,
            pub device: &'a ash::Device,
        }

        impl<'a> $name<'a> {
            pub fn take(mut $sel) -> $ty {
                $sel.drop = false;
                unsafe {
                    std::mem::ManuallyDrop::take(&mut $sel.inner)
                }
            }

            pub fn read(& $sel) -> &$ty {
                &$sel.inner
            }
        }

        impl<'a> Drop for $name<'a> {
            fn drop(&mut $sel) {
                if $sel.drop {
                    unsafe {
                        $drop_fn
                        std::mem::ManuallyDrop::take(&mut $sel.inner);
                    }
                }
            }
        }
    };
}
