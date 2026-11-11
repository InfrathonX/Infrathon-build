"""KVzip KV-scoring evaluation runtime (shared by the public tests and the Judge).

Read-only in the Work image.  Nothing here is task-secret: it implements the
public protocol (prompt, retention rule, masked decoding, metrics, scorer API).
"""
