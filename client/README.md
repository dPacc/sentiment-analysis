# Sentiment Analysis React Client

A simple React client for the Sentiment Analysis API.

## Features

- Text sentiment analysis with visualization
- Real-time sentiment prediction
- Responsive design
- Chart visualization of sentiment probabilities

## Development

### Prerequisites

- Node.js 14+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

### Building for Production

```bash
# Create production build
npm run build
```

## Docker

This project includes a Dockerfile for containerized deployment. To build and run:

```bash
# Build the Docker image
docker build -t sentiment-analysis-client .

# Run the container
docker run -p 3001:80 sentiment-analysis-client
```

## Using with Docker Compose

The client is configured to work with the full sentiment analysis stack using docker-compose. From the root project directory:

```bash
docker-compose up -d
```

The client will be available at <http://localhost:3001/>
