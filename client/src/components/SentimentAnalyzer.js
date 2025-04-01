import React, { useState } from "react";
import { Bar } from "react-chartjs-2";
import { motion, AnimatePresence } from "framer-motion";
import {
  Paper,
  TextField,
  Button,
  Box,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  Fade,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import api from "../api";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

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

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case "positive":
        return "#4caf50";
      case "neutral":
        return "#2196f3";
      case "negative":
        return "#f44336";
      default:
        return "#757575";
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
              "rgba(244, 67, 54, 0.6)",
              "rgba(33, 150, 243, 0.6)",
              "rgba(76, 175, 80, 0.6)",
            ],
            borderColor: [
              "rgba(244, 67, 54, 1)",
              "rgba(33, 150, 243, 1)",
              "rgba(76, 175, 80, 1)",
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
        ticks: {
          callback: (value) => `${(value * 100).toFixed(0)}%`,
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: "Sentiment Analysis Results",
        font: {
          size: 16,
          weight: "bold",
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => `${(context.raw * 100).toFixed(1)}%`,
        },
      },
    },
  };

  return (
    <Box sx={{ width: "100%", maxWidth: 800, mx: "auto" }}>
      <Paper
        component="form"
        onSubmit={handleSubmit}
        elevation={2}
        sx={{
          p: 3,
          borderRadius: 2,
          backgroundColor: "background.paper",
        }}
      >
        <TextField
          fullWidth
          multiline
          rows={4}
          variant="outlined"
          placeholder="Enter your text here to analyze its sentiment..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
          sx={{ mb: 2 }}
        />
        <Box sx={{ display: "flex", justifyContent: "center" }}>
          <Button
            type="submit"
            variant="contained"
            disabled={loading}
            endIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
            sx={{
              minWidth: 200,
              height: 48,
              fontSize: "1.1rem",
            }}
          >
            {loading ? "Analyzing..." : "Analyze Sentiment"}
          </Button>
        </Box>
      </Paper>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          </motion.div>
        )}

        {result && !error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Paper elevation={2} sx={{ mt: 3, p: 3, borderRadius: 2 }}>
              <Typography variant="h6" gutterBottom>
                Analysis Results
              </Typography>

              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <Typography variant="body1" sx={{ mr: 1 }}>
                  Sentiment:
                </Typography>
                <Chip
                  label={result.sentiment.toUpperCase()}
                  sx={{
                    backgroundColor: getSentimentColor(result.sentiment),
                    color: "white",
                    fontWeight: "bold",
                  }}
                />
              </Box>

              <Typography variant="body1" gutterBottom>
                Confidence:{" "}
                <Box
                  component="span"
                  sx={{ fontWeight: "bold", color: "primary.main" }}
                >
                  {(result.confidence * 100).toFixed(2)}%
                </Box>
              </Typography>

              <Box sx={{ height: 300, mt: 3 }}>
                <Bar data={chartData} options={chartOptions} />
              </Box>

              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mt: 2, textAlign: "right" }}
              >
                Processing time: {result.processing_time.toFixed(4)} seconds
              </Typography>
            </Paper>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
};

export default SentimentAnalyzer;
