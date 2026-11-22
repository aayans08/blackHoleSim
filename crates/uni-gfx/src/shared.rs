//! Per-frame inputs the app hands to render features (M1 frame contract, "Shared inputs").
//!
//! The app creates a [`SharedInput`], gives a clone to the feature that needs it, and sets it
//! every frame before `Renderer::render`. Features read it only in `RenderFeature::prepare`, so
//! a frame sees one consistent value.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// A value the app sets and a feature reads. Clones share it.
pub struct SharedInput<T>(Arc<Mutex<T>>);

impl<T> Clone for SharedInput<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> SharedInput<T> {
    /// The value, even if a holder panicked: the value is plain data that each `set` replaces
    /// whole, so a poisoned lock never leaves it half-written.
    fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: Clone> SharedInput<T> {
    pub fn new(v: T) -> Self {
        Self(Arc::new(Mutex::new(v)))
    }

    /// Replaces the value (the app, once per frame, before `Renderer::render`).
    pub fn set(&self, v: T) {
        *self.lock() = v;
    }

    /// A copy of the value.
    pub fn get(&self) -> T {
        self.lock().clone()
    }

    /// Runs `f` on the value without copying it. `f` must not touch this input again (the
    /// lock is held while it runs).
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(&self.lock())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_input_set_get_and_poison() {
        let a = SharedInput::new(vec![1, 2, 3]);
        let b = a.clone();
        assert_eq!(b.get(), vec![1, 2, 3]);
        // A clone shares the value: what the app sets, the feature sees.
        a.set(vec![4]);
        assert_eq!(b.get(), vec![4]);
        assert_eq!(b.with(|v| v.len()), 1);

        // A holder that panics poisons the mutex; get, set and with still work afterwards.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.with(|_| panic!("feature panicked while reading its input"))
        }));
        assert!(caught.is_err());
        assert!(a.0.is_poisoned());
        assert_eq!(a.get(), vec![4]);
        a.set(vec![5, 6]);
        assert_eq!(b.get(), vec![5, 6]);
        assert_eq!(b.with(|v| v.iter().sum::<i32>()), 11);
    }
}
