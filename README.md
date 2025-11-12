# LeetCode Casino 

A gamified LeetCode problem tracker that treats problem-solving as an investment portfolio. Track your progress, maintain streaks, earn tokens, and spin the wheel for rewards!

## Features

- 🎯 **Problem Tracking**: Track solved LeetCode problems with difficulty levels
- 💰 **Investment System**: Earn investment value based on problem difficulty
- 🔥 **Streak System**: Build and maintain daily solving streaks with bonuses
- 🎰 **Reward Wheel**: Earn tokens from solving problems and spin for cash or multipliers
- 📊 **Statistics Dashboard**: View your investment portfolio, streak history, and progress
- 📈 **Leaderboard**: Compete with others on total investment
- ⏰ **Daily Decay**: Investment decays 2% per day without solving (when streak is inactive)
- 🏷️ **Topic Filtering**: Filter problems by topics (Array, Hash Table, etc.)

## How It Works

### Investment Values

- **Easy**: $10 base value
- **Medium**: $25 base value
- **Hard**: $50 base value

### Streak Bonuses

- Each consecutive day adds 4% bonus to investment value
- Example: 5-day streak = 20% bonus (1.2x multiplier)

### Tokens & Rewards

- Solve problems to earn tokens (1 token per problem)
- Use tokens to spin the reward wheel
- Win cash prizes or multipliers to boost your investment

### Daily Decay

- Investment decays 2% per day when you're not maintaining a streak
- Keeps you motivated to solve problems regularly!

## Tech Stack

### Backend

- **Python 3.x**
- **Flask**: RESTful API
- **SQLAlchemy**: ORM for database operations
- **PostgreSQL**: Production database (SQLite for development/testing)
- **LeetCode API**: Fetch problem information

### Frontend

- **React 18**: UI framework
- **Tailwind CSS**: Styling
- **Babel**: JSX transpilation

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL (for production) or SQLite (for development)
- pip

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd Leetcode_Tracker
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

This will install:

- Flask (web framework)
- Flask-CORS (CORS support)
- SQLAlchemy (ORM)
- python-dotenv (environment variables)
- requests (HTTP library)
- leetcode (LeetCode API client)
- psycopg2-binary (PostgreSQL adapter, optional)

3. **Configure environment variables**

Create a `.env` file in the project root:

```env
POSTGRES_URL=postgresql://user:password@localhost/leetcode_tracker
```

For development with SQLite, you can skip this (it will use SQLite by default).

4. **Run the backend server**

```bash
cd src
python main.py
```

The API will be available at `http://localhost:5000`

5. **Open the frontend**

Open `src/index.html` in your browser, or serve it with a local server:

```bash
# Using Python's built-in server
cd src
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## API Endpoints

### Users

- `GET /api/users` - Get all users
- `POST /api/users` - Create a new user
  ```json
  {
    "username": "johndoe",
    "leetcode_username": "johndoe_lc",
    "leetcode_session": "optional_session_token"
  }
  ```
- `DELETE /api/users/<username>` - Delete a user

### User Statistics

- `GET /api/users/<username>/stats` - Get user statistics

### Problems

- `GET /api/users/<username>/problems?limit=10&topic=Array` - Get recent problems
- `POST /api/users/<username>/problems` - Add a solved problem
  ```json
  {
    "problem_slug": "two-sum",
    "difficulty": "Easy" // Optional, will be fetched if not provided
  }
  ```
- `DELETE /api/users/<username>/problems/<problem_id>` - Delete a problem

### Topics

- `GET /api/users/<username>/topics` - Get all topics from solved problems

### History

- `GET /api/users/<username>/history?limit=20` - Get investment history

### Wheel Spinning

- `POST /api/users/<username>/spin` - Spin the reward wheel
  ```json
  {
    "token_type": "Easy" // "Easy", "Medium", or "Hard"
  }
  ```
- `GET /api/users/<username>/spins?limit=10` - Get spin history

### Leaderboard

- `GET /api/leaderboard?limit=10` - Get top users by investment

## Usage Examples

### Adding a Problem

```bash
curl -X POST http://localhost:5000/api/users/johndoe/problems \
  -H "Content-Type: application/json" \
  -d '{"problem_slug": "two-sum"}'
```

### Getting User Stats

```bash
curl http://localhost:5000/api/users/johndoe/stats
```

### Spinning the Wheel

```bash
curl -X POST http://localhost:5000/api/users/johndoe/spin \
  -H "Content-Type: application/json" \
  -d '{"token_type": "Easy"}'
```

## Testing

### Running Tests

1. **Install test dependencies**

```bash
pip install -r requirements-test.txt
```

2. **Run tests**

```bash
pytest src/test_main.py -v
```

3. **Run specific test**

```bash
pytest src/test_main.py::TestStreakLogic::test_multiple_problems_same_day_no_streak_increment -v
```

### Test Coverage

The test suite covers:

- User creation and management
- Streak calculation logic (including bug fixes)
- Problem solving and investment calculation
- Daily decay functionality
- Wheel spinning and token management
- User statistics and leaderboard

See [TEST_README.md](TEST_README.md) for more details.

## Project Structure

```
Leetcode_Tracker/
├── src/
│   ├── main.py              # Flask backend API
│   ├── index.html           # React frontend
│   ├── test_main.py         # Unit tests
│   └── leetcode_tracker.db  # SQLite database (if used)
├── requirements-test.txt    # Test dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── TEST_README.md          # Testing documentation
```

## Configuration

### Investment Values

Edit values in `src/main.py`:

```python
DIFFICULTY_VALUES = {
    'Easy': 10.0,
    'Medium': 25.0,
    'Hard': 50.0
}
```

### Streak Multiplier

```python
STREAK_MULTIPLIER = 0.04  # 4% per day
```

### Daily Decay Rate

```python
DAILY_DECAY_RATE = 0.02  # 2% per day
```

### Prize Ranges

```python
PRIZE_RANGES = {
    'Easy': {
        'cash': (1, 20),
        'multiplier': (0.01, 0.2)
    },
    # ...
}
```

## Database

### Production (PostgreSQL)

Set `POSTGRES_URL` in your `.env` file:

```env
POSTGRES_URL=postgresql://user:password@localhost/dbname
```

### Development (SQLite)

If no `POSTGRES_URL` is set, the application will use SQLite (created automatically).

### Migration

The application automatically handles database migrations. The `problem_slug` column is added automatically if it doesn't exist.

## Known Issues & Limitations

- LeetCode API integration requires a valid session token for authenticated requests
- Problem information fetching uses public LeetCode GraphQL API (no auth needed for basic info)
- SQLite is used for tests (see [TEST_README.md](TEST_README.md) for details)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- LeetCode for the problem platform and API
- Flask and SQLAlchemy communities
- React and Tailwind CSS for the frontend framework

---

**Happy Coding! 🎉**
