from src import progress_log


def test_load_returns_default_schema_when_missing(tmp_path):
    log = progress_log.load(tmp_path)
    assert log["phase"] == "1"
    assert log["last_completed_step"] is None
    assert log["steps"] == []
    assert log["blockers"] == []


def test_save_then_load_roundtrip(tmp_path):
    log = progress_log.load(tmp_path)
    log["next_action"] = "테스트 진행"
    progress_log.save(tmp_path, log)

    reloaded = progress_log.load(tmp_path)
    assert reloaded["next_action"] == "테스트 진행"


def test_record_step_appends_and_updates_last_completed(tmp_path):
    log = progress_log.load(tmp_path)
    log = progress_log.record_step(log, "us_data_collection", "in_progress")
    assert log["last_completed_step"] is None
    assert len(log["steps"]) == 1

    log = progress_log.record_step(log, "us_data_collection", "done", output="ohlcv_meta_us.parquet")
    assert log["last_completed_step"] == "us_data_collection"
    assert len(log["steps"]) == 1
    assert log["steps"][0]["status"] == "done"
    assert log["steps"][0]["output"] == "ohlcv_meta_us.parquet"


def test_save_does_not_leave_tmp_file_on_disk(tmp_path):
    log = progress_log.load(tmp_path)
    progress_log.save(tmp_path, log)

    leftover_tmp_files = list(tmp_path.glob(".progress_log_*.tmp"))
    assert leftover_tmp_files == []
