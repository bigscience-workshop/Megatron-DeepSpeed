.PHONY: test style

check_dirs := tests tools/convert_checkpoint

# this target tests for the library
test:
	pytest tests

# this target runs checks on all files and potentially modifies some of them
style:
	black $(check_dirs)
	isort $(check_dirs)