import React, { useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Create axios instance with base URL
const api = axios.create({
  baseURL: "http://localhost:8080",
  headers: {
    "Content-Type": "application/json",
  },
});

const SentimentAnalyzer = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await api.post("/api/sentiment", {
        text,
      });

      if (response.data.status === "success") {
        setResult(response.data.data);
      } else {
        setError("Error analyzing sentiment: " + response.data.message);
      }
    } catch (err) {
      setError(
        "Error connecting to the API: " + (err.message || "Unknown error")
      );
      console.error("API Error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data if we have results
  const chartData = result
    ? {
        labels: ["Negative", "Neutral", "Positive"],
        datasets: [
          {
            label: "Sentiment Probabilities",
            data: [
              result.probabilities.negative,
              result.probabilities.neutral,
              result.probabilities.positive,
            ],
            backgroundColor: [
              "rgba(255, 99, 132, 0.6)",
              "rgba(54, 162, 235, 0.6)",
              "rgba(75, 192, 192, 0.6)",
            ],
            borderColor: [
              "rgba(255, 99, 132, 1)",
              "rgba(54, 162, 235, 1)",
              "rgba(75, 192, 192, 1)",
            ],
            borderWidth: 1,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: "Sentiment Analysis Results",
      },
    },
  };

  return (
    <div>
      <form className="sentiment-form" onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to analyze sentiment..."
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>
      </form>

      {error && (
        <div className="result-container" style={{ color: "red" }}>
          {error}
        </div>
      )}

      {result && !error && (
        <div className="result-container">
          <h3>Analysis Results</h3>
          <p>
            Sentiment:{" "}
            <span className={result.sentiment}>{result.sentiment}</span>
          </p>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          <div className="chart-container">
            <Bar data={chartData} options={chartOptions} />
          </div>
          <p>Processing time: {result.processing_time.toFixed(4)} seconds</p>
        </div>
      )}
    </div>
  );
};

export default SentimentAnalyzer;
