# Running Tests

This project includes unit tests to verify the functionality of the Leetcode Tracker, especially the streak calculation logic.

## Setup

1. Install test dependencies:

```bash
pip install -r requirements-test.txt
```

Or install pytest directly:

```bash
pip install pytest
```

## Running Tests

From the project root directory:

```bash
# Run all tests
pytest src/test_main.py -v

# Run tests with coverage (if pytest-cov is installed)
pytest src/test_main.py --cov=src/main --cov-report=html -v

# Run a specific test class
pytest src/test_main.py::TestStreakLogic -v

# Run a specific test
pytest src/test_main.py::TestStreakLogic::test_multiple_problems_same_day_no_streak_increment -v
```

## Test Coverage

The test suite covers:

1. **User Creation** - Creating and retrieving users
2. **Streak Logic** - The critical bug fix for streaks not incrementing on same day
   - First problem sets streak to 1
   - Multiple problems on same day don't increment streak
   - Consecutive days increment streak
   - Gaps reset streak
3. **Problem Solving** - Adding problems with difficulty, investment values, tokens
4. **Daily Decay** - Decay application when no streak is active
5. **Wheel Spinning** - Token deduction and prize distribution
6. **User Statistics** - Retrieving user stats
7. **Problem Deletion** - Removing problems and adjusting investment

## Key Test: Streak Bug Fix

The most important test is `test_multiple_problems_same_day_no_streak_increment` which verifies that:

- Adding multiple problems on the same day does NOT increment the streak
- This was the bug that was fixed in the codebase

All tests use an in-memory SQLite database, so they don't affect your actual database.
