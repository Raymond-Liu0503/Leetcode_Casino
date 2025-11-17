"""
Unit tests for Leetcode Investment Tracker
"""

import os
import sys
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    DAILY_DECAY_RATE,
    DIFFICULTY_TOKENS,
    DIFFICULTY_VALUES,
    STREAK_MULTIPLIER,
    InvestmentHistory,
    LeetcodeTracker,
    SolvedProblem,
    User,
    WheelSpin,
)


@pytest.fixture
def tracker():
    """Create a tracker instance with in-memory database for testing"""
    # Use in-memory SQLite database for testing
    test_db_url = "sqlite:///:memory:"
    tracker = LeetcodeTracker(database_url=test_db_url)
    yield tracker
    # Cleanup
    tracker.session.close()


@pytest.fixture
def user(tracker):
    """Create a test user"""
    return tracker.create_user("test@example.com", "password123", "testuser", "testuser_leetcode")


class TestUserCreation:
    """Tests for user creation"""

    def test_create_user(self, tracker):
        """Test creating a new user"""
        user = tracker.create_user("newuser@example.com", "password123", "newuser", "newuser_leetcode")
        assert user.username == "newuser"
        assert user.leetcode_username == "newuser_leetcode"
        assert user.total_investment == 0.0
        assert user.current_streak == 0
        assert user.max_streak == 0

    def test_get_user(self, tracker, user):
        """Test retrieving a user"""
        retrieved_user = tracker.get_user("testuser")
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"

    def test_get_nonexistent_user(self, tracker):
        """Test retrieving a nonexistent user"""
        retrieved_user = tracker.get_user("nonexistent")
        assert retrieved_user is None


class TestStreakLogic:
    """Tests for streak calculation - especially the bug fix"""

    def test_first_problem_sets_streak_to_one(self, tracker, user):
        """Test that solving the first problem sets streak to 1"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            result = tracker.add_solved_problem("testuser", "test-problem")
            assert result is not None
            user = tracker.get_user("testuser")
            assert user.current_streak == 1
            assert user.max_streak == 1

    def test_multiple_problems_same_day_no_streak_increment(self, tracker, user):
        """Test that adding multiple problems on the same day doesn't increment streak"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            # Add first problem
            result1 = tracker.add_solved_problem("testuser", "test-problem-1")
            assert result1 is not None
            user = tracker.get_user("testuser")
            streak_after_first = user.current_streak
            assert streak_after_first == 1

            # Add second problem on same day
            result2 = tracker.add_solved_problem("testuser", "test-problem-2")
            assert result2 is not None
            user = tracker.get_user("testuser")
            streak_after_second = user.current_streak
            # Streak should NOT increment - this was the bug!
            assert streak_after_second == streak_after_first
            assert streak_after_second == 1

    def test_consecutive_days_increment_streak(self, tracker, user):
        """Test that solving on consecutive days increments the streak"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            # Solve problem on day 1
            result1 = tracker.add_solved_problem("testuser", "test-problem-1")
            user = tracker.get_user("testuser")
            assert user.current_streak == 1

            # Get the first problem and update its solved_at to yesterday
            # Also update last_solved_date to yesterday
            yesterday = datetime.now(UTC) - timedelta(days=1)
            problem1 = tracker.session.query(SolvedProblem).filter_by(
                user_id=user.id,
                problem_slug="test-problem-1"
            ).first()
            problem1.solved_at = yesterday
            user.last_solved_date = yesterday
            tracker.session.commit()

            # Solve problem on day 2 (today)
            tracker.add_solved_problem("testuser", "test-problem-2")
            user = tracker.get_user("testuser")
            assert user.current_streak == 2
            assert user.max_streak == 2

    def test_gap_in_solving_resets_streak(self, tracker, user):
        """Test that a gap in solving resets the streak to 1"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            # Solve problem on day 1
            result1 = tracker.add_solved_problem("testuser", "test-problem-1")
            user = tracker.get_user("testuser")
            assert user.current_streak == 1

            # Simulate a gap of 2 days - update both problem timestamp and last_solved_date
            two_days_ago = datetime.now(UTC) - timedelta(days=2)
            problem1 = tracker.session.query(SolvedProblem).filter_by(
                user_id=user.id,
                problem_slug="test-problem-1"
            ).first()
            problem1.solved_at = two_days_ago
            user.last_solved_date = two_days_ago
            # Keep streak at 1 so apply_daily_decay can reset it to 0
            # (apply_daily_decay resets streak if > 0 and last_solved > 1 day ago)
            tracker.session.commit()

            # Solve problem after gap (today)
            # apply_daily_decay will reset streak to 0, then add_solved_problem will set it to 1
            tracker.add_solved_problem("testuser", "test-problem-2")
            user = tracker.get_user("testuser")
            # Streak should reset to 1, not increment
            assert user.current_streak == 1

    def test_streak_bonus_calculation(self, tracker):
        """Test that streak bonus is calculated correctly"""
        # Test with different streak values
        base_value = 10.0
        assert tracker.calculate_streak_bonus(base_value, 0) == 10.0
        assert tracker.calculate_streak_bonus(base_value, 1) == 10.0 * (1 + (1 * STREAK_MULTIPLIER))
        assert tracker.calculate_streak_bonus(base_value, 5) == 10.0 * (1 + (5 * STREAK_MULTIPLIER))
        assert tracker.calculate_streak_bonus(base_value, 10) == 10.0 * (1 + (10 * STREAK_MULTIPLIER))


class TestProblemSolving:
    """Tests for adding solved problems"""

    def test_add_problem_with_difficulty(self, tracker, user):
        """Test adding a problem with specified difficulty"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Two Sum',
            'difficulty': 'Easy',
            'topics': ['Array', 'Hash Table']
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            result = tracker.add_solved_problem("testuser", "two-sum")
            assert result is not None
            assert result['problem']['difficulty'] == "Easy"
            assert result['problem']['investment_value'] > 0

    def test_add_problem_fetches_difficulty(self, tracker, user):
        """Test that difficulty is fetched if not provided"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Two Sum',
            'difficulty': 'Medium',
            'topics': ['Array']
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            result = tracker.add_solved_problem("testuser", "two-sum")
            assert result is not None
            assert result['problem']['difficulty'] == "Medium"

    def test_investment_value_includes_streak_bonus(self, tracker, user):
        """Test that investment value includes streak bonus"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            # Set up a streak by having solved a problem yesterday
            # First, solve a problem and update it to be from yesterday
            result1 = tracker.add_solved_problem("testuser", "test-problem-1")
            yesterday = datetime.now(UTC) - timedelta(days=1)
            problem1 = tracker.session.query(SolvedProblem).filter_by(
                user_id=user.id,
                problem_slug="test-problem-1"
            ).first()
            problem1.solved_at = yesterday
            user.last_solved_date = yesterday
            user.current_streak = 5  # Set streak to 5 (simulating 5 consecutive days)
            user.total_multiplier = 1.0  # Reset multiplier for clean test
            tracker.session.commit()

            # Now solve a problem today - streak should increment to 6
            result = tracker.add_solved_problem("testuser", "test-problem-2")
            base_value = DIFFICULTY_VALUES['Easy']
            # Streak will be 6 after incrementing from 5
            expected_value = base_value * (1 + (6 * STREAK_MULTIPLIER))

            assert result['problem']['investment_value'] == pytest.approx(expected_value, rel=0.01)

    def test_tokens_earned(self, tracker, user):
        """Test that correct number of tokens are earned"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            tracker.add_solved_problem("testuser", "test-problem")
            user = tracker.get_user("testuser")
            assert user.tokens_easy == 1
            assert user.tokens_medium == 0
            assert user.tokens_hard == 0


class TestDailyDecay:
    """Tests for daily decay functionality"""

    def test_no_decay_when_streak_active(self, tracker, user):
        """Test that decay doesn't apply when user has an active streak"""
        user.total_investment = 100.0
        user.current_streak = 5
        user.last_solved_date = datetime.now(UTC) - timedelta(days=1)
        tracker.session.commit()

        initial_investment = user.total_investment
        tracker.apply_daily_decay(user)
        user = tracker.get_user("testuser")

        # Investment should not decay when streak is active
        assert user.total_investment == initial_investment

    def test_decay_applies_when_no_streak(self, tracker, user):
        """Test that decay applies when user has no streak"""
        user.total_investment = 100.0
        user.current_streak = 0
        user.last_solved_date = datetime.now(UTC) - timedelta(days=2)
        tracker.session.commit()

        initial_investment = user.total_investment
        tracker.apply_daily_decay(user)
        user = tracker.get_user("testuser")

        # Investment should decay
        assert user.total_investment < initial_investment

    def test_streak_broken_after_missing_day(self, tracker, user):
        """Test that streak is broken if user missed a day"""
        user.current_streak = 5
        user.last_solved_date = datetime.now(UTC) - timedelta(days=2)
        tracker.session.commit()

        tracker.apply_daily_decay(user)
        user = tracker.get_user("testuser")

        # Streak should be broken (set to 0)
        assert user.current_streak == 0


class TestWheelSpin:
    """Tests for wheel spinning functionality"""

    def test_spin_wheel_requires_token(self, tracker, user):
        """Test that spinning requires a token"""
        user.tokens_easy = 0
        tracker.session.commit()

        result = tracker.spin_wheel("testuser", "Easy")
        assert result is None

    def test_spin_wheel_deducts_token(self, tracker, user):
        """Test that spinning deducts a token"""
        user.tokens_easy = 1
        tracker.session.commit()

        with patch('random.random', return_value=0.5):  # 50% - will get cash
            with patch('random.uniform', return_value=10.0):
                result = tracker.spin_wheel("testuser", "Easy")
                assert result is not None
                user = tracker.get_user("testuser")
                assert user.tokens_easy == 0

    def test_spin_wheel_cash_prize(self, tracker, user):
        """Test that cash prize increases investment"""
        user.tokens_easy = 1
        user.total_investment = 100.0
        tracker.session.commit()

        initial_investment = user.total_investment

        with patch('random.random', return_value=0.5):  # Cash spin
            with patch('random.uniform', return_value=25.0):
                result = tracker.spin_wheel("testuser", "Easy")
                assert result is not None
                assert result['spin_type'] == 'cash'
                user = tracker.get_user("testuser")
                assert user.total_investment == initial_investment + 25.0


class TestUserStats:
    """Tests for user statistics"""

    def test_get_user_stats(self, tracker, user):
        """Test retrieving user statistics"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            tracker.add_solved_problem("testuser", "test-problem")
            stats = tracker.get_user_stats("testuser")

            assert stats is not None
            assert stats['username'] == "testuser"
            assert stats['total_problems'] == 1
            assert stats['current_streak'] == 1
            assert 'total_investment' in stats
            assert 'tokens' in stats


class TestProblemDeletion:
    """Tests for deleting problems"""

    def test_delete_problem(self, tracker, user):
        """Test deleting a problem"""
        mock_service_instance = Mock()
        mock_service_instance.get_problem_info.return_value = {
            'questionId': '1',
            'title': 'Test Problem',
            'difficulty': 'Easy',
            'topics': []
        }

        with patch('main.LeetcodeService', return_value=mock_service_instance):
            result = tracker.add_solved_problem("testuser", "test-problem")
            problem_id = result['problem']['id']
            initial_investment = tracker.get_user("testuser").total_investment

            success = tracker.delete_problem("testuser", problem_id)
            assert success is True

            user = tracker.get_user("testuser")
            assert user.total_investment < initial_investment
            assert user.tokens_easy == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

