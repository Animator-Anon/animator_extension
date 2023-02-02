import launch

if not launch.is_installed("glob"):
    launch.run_pip("install glob", "requirements for AnimatorAnon Extension")
