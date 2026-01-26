"""
Unit tests for validate_deployment.py script.

Tests cover:
- Successful validation scenarios
- Validation failures (model and database)
- Missing configuration handling
- Verbose output
- Exit codes
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_deployment import (
    DeploymentValidator,
    load_config_file,
    parse_checksums,
    load_configuration,
    main,
)
from app.services.startup_validation import ValidationResult


class TestDeploymentValidator:
    """Test DeploymentValidator class"""

    def test_init_with_defaults(self):
        """Test validator initialization with default values"""
        validator = DeploymentValidator()
        assert validator.model_validator is not None
        assert validator.db_validator is not None
        assert validator.verbose is False

    def test_init_with_custom_values(self):
        """Test validator initialization with custom values"""
        validator = DeploymentValidator(
            model_paths=["/path/to/model.pkl"],
            model_checksums={"/path/to/model.pkl": "abc123"},
            database_url="postgresql://test:test@localhost:5432/test",
            db_timeout=5.0,
            model_timeout=10.0,
            verbose=True,
        )
        assert validator.verbose is True

    @patch("scripts.validate_deployment.ModelValidator")
    @patch("scripts.validate_deployment.DatabaseValidator")
    def test_validate_all_success(self, mock_db_validator, mock_model_validator):
        """Test successful validation of all components"""
        # Setup mocks
        model_result = ValidationResult("ML Models", "healthy")
        db_result = ValidationResult("Database", "healthy")

        mock_model_validator_instance = Mock()
        mock_model_validator_instance.validate.return_value = model_result
        mock_model_validator.return_value = mock_model_validator_instance

        mock_db_validator_instance = Mock()
        mock_db_validator_instance.validate.return_value = db_result
        mock_db_validator.return_value = mock_db_validator_instance

        # Test
        validator = DeploymentValidator()
        all_healthy, results = validator.validate_all()

        # Assertions
        assert all_healthy is True
        assert len(results) == 2
        assert all(r.is_healthy() for r in results)

    @patch("scripts.validate_deployment.ModelValidator")
    @patch("scripts.validate_deployment.DatabaseValidator")
    def test_validate_all_model_failure(self, mock_db_validator, mock_model_validator):
        """Test validation failure when model validation fails"""
        # Setup mocks
        model_result = ValidationResult("ML Models", "unhealthy", "Model file not found")
        db_result = ValidationResult("Database", "healthy")

        mock_model_validator_instance = Mock()
        mock_model_validator_instance.validate.return_value = model_result
        mock_model_validator.return_value = mock_model_validator_instance

        mock_db_validator_instance = Mock()
        mock_db_validator_instance.validate.return_value = db_result
        mock_db_validator.return_value = mock_db_validator_instance

        # Test
        validator = DeploymentValidator()
        all_healthy, results = validator.validate_all()

        # Assertions
        assert all_healthy is False
        assert len(results) == 2
        assert not results[0].is_healthy()
        assert results[1].is_healthy()

    @patch("scripts.validate_deployment.ModelValidator")
    @patch("scripts.validate_deployment.DatabaseValidator")
    def test_validate_all_database_failure(self, mock_db_validator, mock_model_validator):
        """Test validation failure when database validation fails"""
        # Setup mocks
        model_result = ValidationResult("ML Models", "healthy")
        db_result = ValidationResult("Database", "unhealthy", "Connection refused")

        mock_model_validator_instance = Mock()
        mock_model_validator_instance.validate.return_value = model_result
        mock_model_validator.return_value = mock_model_validator_instance

        mock_db_validator_instance = Mock()
        mock_db_validator_instance.validate.return_value = db_result
        mock_db_validator.return_value = mock_db_validator_instance

        # Test
        validator = DeploymentValidator()
        all_healthy, results = validator.validate_all()

        # Assertions
        assert all_healthy is False
        assert len(results) == 2
        assert results[0].is_healthy()
        assert not results[1].is_healthy()

    def test_format_report_success(self):
        """Test report formatting for successful validation"""
        results = [
            ValidationResult("ML Models", "healthy"),
            ValidationResult("Database", "healthy"),
        ]
        validator = DeploymentValidator()
        report = validator.format_report(results)

        assert "PRE-DEPLOYMENT VALIDATION REPORT" in report
        assert "READY FOR DEPLOYMENT" in report
        assert "✓" in report
        assert "✗" not in report

    def test_format_report_failure(self):
        """Test report formatting for failed validation"""
        results = [
            ValidationResult("ML Models", "unhealthy", "Model file not found"),
            ValidationResult("Database", "healthy"),
        ]
        validator = DeploymentValidator()
        report = validator.format_report(results)

        assert "PRE-DEPLOYMENT VALIDATION REPORT" in report
        assert "NOT READY FOR DEPLOYMENT" in report
        assert "✗" in report
        assert "Model file not found" in report

    def test_get_fix_instructions_model_not_found(self):
        """Test fix instructions for missing model files"""
        result = ValidationResult("ML Models", "unhealthy", "Model file not found")
        validator = DeploymentValidator()
        fix = validator._get_fix_instructions(result)

        assert fix is not None
        assert "train_ml_models.py" in fix.lower()

    def test_get_fix_instructions_database_connection_refused(self):
        """Test fix instructions for database connection issues"""
        result = ValidationResult("Database", "unhealthy", "Connection refused")
        validator = DeploymentValidator()
        fix = validator._get_fix_instructions(result)

        assert fix is not None
        assert "database" in fix.lower() or "DATABASE_URL" in fix


class TestConfigLoading:
    """Test configuration loading functions"""

    def test_load_config_file_success(self):
        """Test loading valid config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "model_paths": ["/path/to/model.pkl"],
                "db_timeout": 5.0,
            }
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config_file(config_path)
            assert config["model_paths"] == ["/path/to/model.pkl"]
            assert config["db_timeout"] == 5.0
        finally:
            os.unlink(config_path)

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file"""
        with pytest.raises(SystemExit):
            load_config_file("/nonexistent/path/config.json")

    def test_load_config_file_invalid_json(self):
        """Test loading invalid JSON config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name

        try:
            with pytest.raises(SystemExit):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)

    def test_parse_checksums_valid(self):
        """Test parsing valid checksum string"""
        checksum_string = "path1:checksum1,path2:checksum2"
        checksums = parse_checksums(checksum_string)

        assert checksums["path1"] == "checksum1"
        assert checksums["path2"] == "checksum2"

    def test_parse_checksums_empty(self):
        """Test parsing empty checksum string"""
        checksums = parse_checksums("")
        assert checksums == {}

    def test_parse_checksums_invalid_format(self):
        """Test parsing invalid checksum format"""
        checksums = parse_checksums("invalid_format")
        assert checksums == {}

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost:5432/test"})
    def test_load_configuration_from_env(self):
        """Test loading configuration from environment variables"""
        args = Mock()
        args.config = None
        args.model_paths = None
        args.model_checksums = None
        args.database_url = None
        args.db_timeout = None
        args.model_timeout = None

        config = load_configuration(args)
        assert config["database_url"] == "postgresql://test:test@localhost:5432/test"

    @patch.dict(os.environ, {}, clear=True)
    def test_load_configuration_missing_database_url(self):
        """Test loading configuration when DATABASE_URL is missing"""
        args = Mock()
        args.config = None
        args.model_paths = None
        args.model_checksums = None
        args.database_url = None
        args.db_timeout = None
        args.model_timeout = None

        config = load_configuration(args)
        assert config["database_url"] is None


class TestMainFunction:
    """Test main() function and exit codes"""

    @patch("scripts.validate_deployment.DeploymentValidator")
    @patch("scripts.validate_deployment.load_configuration")
    @patch("sys.argv", ["validate_deployment.py", "--verbose"])
    def test_main_success(self, mock_load_config, mock_validator_class):
        """Test main() with successful validation"""
        # Setup mocks
        mock_config = {
            "model_paths": None,
            "model_checksums": None,
            "database_url": "postgresql://test:test@localhost:5432/test",
            "db_timeout": 2.0,
            "model_timeout": None,
        }
        mock_load_config.return_value = mock_config

        mock_validator = Mock()
        mock_validator.validate_all.return_value = (
            True,
            [
                ValidationResult("ML Models", "healthy"),
                ValidationResult("Database", "healthy"),
            ],
        )
        mock_validator.format_report.return_value = "Test Report"
        mock_validator_class.return_value = mock_validator

        # Test
        exit_code = main()
        assert exit_code == 0

    @patch("scripts.validate_deployment.DeploymentValidator")
    @patch("scripts.validate_deployment.load_configuration")
    @patch("sys.argv", ["validate_deployment.py"])
    def test_main_failure(self, mock_load_config, mock_validator_class):
        """Test main() with failed validation"""
        # Setup mocks
        mock_config = {
            "model_paths": None,
            "model_checksums": None,
            "database_url": "postgresql://test:test@localhost:5432/test",
            "db_timeout": 2.0,
            "model_timeout": None,
        }
        mock_load_config.return_value = mock_config

        mock_validator = Mock()
        mock_validator.validate_all.return_value = (
            False,
            [
                ValidationResult("ML Models", "unhealthy", "Model not found"),
                ValidationResult("Database", "healthy"),
            ],
        )
        mock_validator.format_report.return_value = "Test Report"
        mock_validator_class.return_value = mock_validator

        # Test
        exit_code = main()
        assert exit_code == 1

    @patch("scripts.validate_deployment.load_configuration")
    @patch("sys.argv", ["validate_deployment.py"])
    @patch.dict(os.environ, {}, clear=True)
    def test_main_missing_database_url(self, mock_load_config):
        """Test main() when DATABASE_URL is missing"""
        mock_config = {
            "model_paths": None,
            "model_checksums": None,
            "database_url": None,
            "db_timeout": 2.0,
            "model_timeout": None,
        }
        mock_load_config.return_value = mock_config

        exit_code = main()
        assert exit_code == 1
