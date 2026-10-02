//! Own test binary: a recorded device failure is sticky for the whole process.
#[test]
fn device_thread_panics_fail_later_checks_but_other_threads_do_not() {
    ancha::device::install_guard();
    ancha::device::install_guard();
    let worker = std::thread::Builder::new()
        .name("worker".into())
        .spawn(|| panic!("ordinary worker"))
        .unwrap();
    assert!(worker.join().is_err());
    ancha::device::check().unwrap();
    let server = std::thread::Builder::new()
        .name("DSD-0-0".into())
        .spawn(|| panic!("can't allocate buffer of size: 42"))
        .unwrap();
    assert!(server.join().is_err());
    let error = ancha::device::check().unwrap_err().to_string();
    assert!(
        error.contains("can't allocate buffer of size: 42"),
        "{error}"
    );
    assert!(ancha::device::check().is_err());
}
