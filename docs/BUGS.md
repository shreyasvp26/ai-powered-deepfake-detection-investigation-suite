# Bug and known-limitation tracker

| ID | Description | Severity | Status | Fix / notes |
|----|---------------|----------|--------|-------------|
| BUG-001 | `DSANGradCAMWrapper._srm` is not thread-safe under Flask `threaded=True` — concurrent requests can corrupt SRM tensors (FIX-8). | Medium | Known limitation | Acceptable for single-user BTech demo. For production or concurrent demos: new wrapper per request or request-scoped locking. See project plan Sections 10.9 and 13. |
