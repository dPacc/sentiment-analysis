import React, { useState, useRef } from "react";
import {
  Paper,
  TextField,
  Button,
  Box,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
} from "@mui/material";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { CloudUpload, ContentPaste, AutoAwesome } from "@mui/icons-material";
import { faker } from "@faker-js/faker";
import api from "../api";

const BulkAnalyzer = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [texts, setTexts] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loadTestConfig, setLoadTestConfig] = useState({
    users: 10,
    duration: 30,
    rampUp: 5,
    batchSize: 5,
    thinkTime: 1.0,
  });
  const [loadTestResults, setLoadTestResults] = useState(null);
  const fileInputRef = useRef();

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setTexts(e.target.result);
      };
      reader.readAsText(file);
    }
  };

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      setTexts(text);
    } catch (err) {
      setError("Failed to paste from clipboard");
    }
  };

  const generateSampleData = () => {
    const sampleSize = 100;
    const generatedTexts = Array(sampleSize)
      .fill()
      .map(() => {
        const sentiment =
          Math.random() < 0.33
            ? "negative"
            : Math.random() < 0.66
            ? "neutral"
            : "positive";

        let text = "";
        switch (sentiment) {
          case "positive":
            text = faker.helpers.arrayElement([
              `Love the ${faker.commerce.productName()}!`,
              `Great experience with ${faker.company.name()}`,
              `Excellent service and ${faker.commerce.productAdjective()} quality`,
            ]);
            break;
          case "negative":
            text = faker.helpers.arrayElement([
              `Disappointed with the ${faker.commerce.productName()}`,
              `Poor service from ${faker.company.name()}`,
              `Would not recommend their ${faker.commerce.product()}`,
            ]);
            break;
          default:
            text = faker.helpers.arrayElement([
              `Average ${faker.commerce.productName()}`,
              `Neutral opinion about ${faker.company.name()}`,
              `It's okay but ${faker.commerce.productAdjective()}`,
            ]);
        }
        return text;
      });

    setTexts(generatedTexts.join("\n"));
  };

  const analyzeBulk = async () => {
    setLoading(true);
    setError(null);
    try {
      const textArray = texts.split("\n").filter((text) => text.trim());
      const response = await api.post("/api/batch-sentiment", {
        texts: textArray,
      });

      if (response.data.status === "success") {
        setResults(response.data.data);
      } else {
        throw new Error(response.data.message || "Failed to analyze texts");
      }
    } catch (err) {
      setError(err.message || "Failed to analyze texts");
      console.error("API Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const runLoadTest = async () => {
    setLoading(true);
    setError(null);
    try {
      const textArray = texts.split("\n").filter((text) => text.trim());
      const response = await api.post("/api/load-test", {
        texts: textArray,
        ...loadTestConfig,
      });

      if (response.data.status === "success") {
        setLoadTestResults(response.data.data);
      } else {
        throw new Error(response.data.message || "Load test failed");
      }
    } catch (err) {
      setError(err.message || "Load test failed");
      console.error("API Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const renderAnalysisResults = () => {
    if (!results) return null;

    const sentimentCounts = results.reduce((acc, r) => {
      acc[r.sentiment] = (acc[r.sentiment] || 0) + 1;
      return acc;
    }, {});

    const chartData = Object.entries(sentimentCounts).map(
      ([sentiment, count]) => ({
        sentiment,
        count,
        percentage: ((count / results.length) * 100).toFixed(1),
      })
    );

    return (
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Analysis Results
        </Typography>
        <Paper sx={{ p: 2, mb: 2 }}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="sentiment" />
              <YAxis />
              <ChartTooltip />
              <Legend />
              <Bar dataKey="count" fill="#8884d8" name="Count" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>

        <Box sx={{ mt: 2 }}>
          {results.map((result, index) => (
            <Paper key={index} sx={{ p: 2, mb: 1 }}>
              <Typography variant="body2" color="text.secondary">
                {result.text}
              </Typography>
              <Box sx={{ mt: 1, display: "flex", gap: 1 }}>
                <Chip
                  label={result.sentiment}
                  color={
                    result.sentiment === "positive"
                      ? "success"
                      : result.sentiment === "negative"
                      ? "error"
                      : "default"
                  }
                  size="small"
                />
                <Chip
                  label={`${(result.confidence * 100).toFixed(1)}% confidence`}
                  variant="outlined"
                  size="small"
                />
              </Box>
            </Paper>
          ))}
        </Box>
      </Box>
    );
  };

  const renderLoadTestResults = () => {
    if (!loadTestResults) return null;

    return (
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Load Test Results - How did the API do?
        </Typography>

        <Paper sx={{ p: 2, mb: 2, backgroundColor: "#f9f9f9" }}>
          <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
            Summary (The Quick Look)
          </Typography>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: 2,
              mt: 2,
            }}
          >
            <Box>
              <Typography variant="body2" color="text.secondary">
                Total Requests
              </Typography>
              <Tooltip title="How many times we asked the API for an answer during the test.">
                <Typography variant="h6">
                  {loadTestResults.total_requests}
                </Typography>
              </Tooltip>
              <Typography variant="caption" color="text.secondary">
                Total asks sent
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Success Rate
              </Typography>
              <Tooltip title="Out of all the asks, how many got a proper answer back? (Higher is better!) ">
                <Typography variant="h6">
                  {loadTestResults.success_rate.toFixed(1)}%
                </Typography>
              </Tooltip>
              <Typography variant="caption" color="text.secondary">
                Successful answers %
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Avg Response Time
              </Typography>
              <Tooltip title="On average, how long did it take to get an answer back? (Lower is better!) ">
                <Typography variant="h6">
                  {loadTestResults.response_time.mean.toFixed(3)}s
                </Typography>
              </Tooltip>
              <Typography variant="caption" color="text.secondary">
                Average wait time (seconds)
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Throughput
              </Typography>
              <Tooltip title="How many answers could the API give each second? (Higher is better!) ">
                <Typography variant="h6">
                  {loadTestResults.throughput.toFixed(1)} req/s
                </Typography>
              </Tooltip>
              <Typography variant="caption" color="text.secondary">
                Answers per second
              </Typography>
            </Box>
          </Box>
        </Paper>

        <Paper sx={{ p: 2, mb: 2, backgroundColor: "#f9f9f9" }}>
          <Typography
            variant="subtitle1"
            sx={{ fontWeight: "bold" }}
            gutterBottom
          >
            Response Time Distribution
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This chart shows how quickly the API answered over time. The lower
            the lines, the faster it was!
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={loadTestResults.response_time_series}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                label={{
                  value: "Time (seconds into test)",
                  position: "insideBottom",
                  offset: -5,
                }}
              />
              <YAxis
                label={{
                  value: "Response Time (s)",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <ChartTooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="mean"
                stroke="#8884d8"
                name="Average Time"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="p95"
                stroke="#82ca9d"
                name="95% Slowest Time"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mt: 1, display: "block" }}
          >
            'Average Time' is the usual wait time. '95% Slowest Time' shows the
            wait time for almost all requests (we ignore the very slowest 5%).
          </Typography>
        </Paper>

        {/* Optional: Add sentiment distribution chart if needed */}
        {loadTestResults.sentiment_distribution && (
          <Paper sx={{ p: 2, mb: 2, backgroundColor: "#f9f9f9" }}>
            <Typography
              variant="subtitle1"
              sx={{ fontWeight: "bold" }}
              gutterBottom
            >
              Sentiment Found During Test
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              What kind of feelings (positive, negative, neutral) did the API
              find in the texts during the test?
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={Object.entries(
                  loadTestResults.sentiment_distribution
                ).map(([key, value]) => ({ name: key, count: value.count }))}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <ChartTooltip />
                <Legend />
                <Bar dataKey="count" fill="#ffc658" name="Number of texts" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        )}

        {loadTestResults.errors && loadTestResults.errors.length > 0 && (
          <Paper sx={{ p: 2, backgroundColor: "#fff3f3" }}>
            <Typography
              variant="subtitle1"
              color="error"
              sx={{ fontWeight: "bold" }}
            >
              Errors ({loadTestResults.errors.length})
            </Typography>
            <Typography variant="body2" color="error.dark" sx={{ mb: 1 }}>
              Oops! The API had some trouble answering these times.
            </Typography>
            <Box
              sx={{
                maxHeight: 150,
                overflowY: "auto",
                fontSize: "0.8rem",
                p: 1,
                border: "1px solid #ffcccc",
                borderRadius: 1,
              }}
            >
              {loadTestResults.errors.map((err, index) => (
                <div key={index}>
                  [{err.timestamp.toFixed(2)}s] {err.error}
                </div>
              ))}
            </Box>
          </Paper>
        )}
      </Box>
    );
  };

  return (
    <Paper sx={{ p: 3, mt: 3 }}>
      <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Bulk Analysis" />
        <Tab label="Load Testing" />
      </Tabs>

      <Box sx={{ mb: 3 }}>
        <Box sx={{ mb: 2, display: "flex", gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<CloudUpload />}
            onClick={() => fileInputRef.current.click()}
          >
            Upload File
          </Button>
          <Button
            variant="outlined"
            startIcon={<ContentPaste />}
            onClick={handlePaste}
          >
            Paste
          </Button>
          <Button
            variant="outlined"
            startIcon={<AutoAwesome />}
            onClick={generateSampleData}
          >
            Generate Sample Data
          </Button>
        </Box>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          style={{ display: "none" }}
          accept=".txt"
        />

        <TextField
          multiline
          rows={8}
          fullWidth
          value={texts}
          onChange={(e) => setTexts(e.target.value)}
          placeholder="Enter multiple texts, one per line..."
          variant="outlined"
        />
      </Box>

      {activeTab === 0 ? (
        <Button
          variant="contained"
          onClick={analyzeBulk}
          disabled={!texts.trim() || loading}
          fullWidth
        >
          {loading ? <CircularProgress size={24} /> : "Analyze Texts"}
        </Button>
      ) : (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Load Test Configuration
          </Typography>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: 2,
              mb: 2,
            }}
          >
            <Tooltip title="Number of simulated users making requests concurrently.">
              <FormControl>
                <InputLabel>Concurrent Users</InputLabel>
                <Select
                  value={loadTestConfig.users}
                  onChange={(e) =>
                    setLoadTestConfig((prev) => ({
                      ...prev,
                      users: e.target.value,
                    }))
                  }
                  label="Concurrent Users"
                >
                  {[5, 10, 20, 50, 100].map((n) => (
                    <MenuItem key={n} value={n}>
                      {n}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Tooltip>

            <Tooltip title="Total duration of the load test in seconds.">
              <FormControl>
                <InputLabel>Duration (seconds)</InputLabel>
                <Select
                  value={loadTestConfig.duration}
                  onChange={(e) =>
                    setLoadTestConfig((prev) => ({
                      ...prev,
                      duration: e.target.value,
                    }))
                  }
                  label="Duration (seconds)"
                >
                  {[10, 30, 60, 120, 300].map((n) => (
                    <MenuItem key={n} value={n}>
                      {n}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Tooltip>

            <Tooltip title="Number of texts sent in each batch request (1 for single requests).">
              <FormControl>
                <InputLabel>Batch Size</InputLabel>
                <Select
                  value={loadTestConfig.batchSize}
                  onChange={(e) =>
                    setLoadTestConfig((prev) => ({
                      ...prev,
                      batchSize: e.target.value,
                    }))
                  }
                  label="Batch Size"
                >
                  {[1, 5, 10, 20, 50].map((n) => (
                    <MenuItem key={n} value={n}>
                      {n}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Tooltip>

            <Tooltip title="Average pause time (in seconds) between requests for each simulated user.">
              <FormControl>
                <InputLabel>Think Time (seconds)</InputLabel>
                <Select
                  value={loadTestConfig.thinkTime}
                  onChange={(e) =>
                    setLoadTestConfig((prev) => ({
                      ...prev,
                      thinkTime: e.target.value,
                    }))
                  }
                  label="Think Time (seconds)"
                >
                  {[0.1, 0.5, 1.0, 2.0, 5.0].map((n) => (
                    <MenuItem key={n} value={n}>
                      {n}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Tooltip>
          </Box>

          <Button
            variant="contained"
            onClick={runLoadTest}
            disabled={!texts.trim() || loading}
            fullWidth
          >
            {loading ? <CircularProgress size={24} /> : "Run Load Test"}
          </Button>
        </Box>
      )}

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          </motion.div>
        )}
      </AnimatePresence>

      {activeTab === 0 ? renderAnalysisResults() : renderLoadTestResults()}
    </Paper>
  );
};

export default BulkAnalyzer;
