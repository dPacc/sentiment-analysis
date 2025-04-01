import React, { useState } from "react";
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Container,
  Typography,
  Box,
  Tabs,
  Tab,
  Paper,
} from "@mui/material";
import { motion } from "framer-motion";
import SentimentAnalyzer from "./components/SentimentAnalyzer";
import BulkAnalyzer from "./components/BulkAnalyzer";
import "./App.css";

// Create a custom theme
const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#2E3B55",
    },
    secondary: {
      main: "#FF6B6B",
    },
    background: {
      default: "#f5f5f7",
      paper: "#ffffff",
    },
  },
  typography: {
    fontFamily: '"Inter", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: "2.5rem",
      fontWeight: 700,
      letterSpacing: "-0.01562em",
    },
    h4: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          borderRadius: 8,
        },
      },
    },
  },
});

function App() {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}
      >
        <AppBar
          position="static"
          elevation={0}
          sx={{ backgroundColor: "background.paper" }}
        >
          <Container maxWidth="lg">
            <Toolbar disableGutters sx={{ height: 80 }}>
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Typography
                  variant="h6"
                  component="div"
                  sx={{
                    flexGrow: 1,
                    color: "primary.main",
                    fontWeight: 700,
                    fontSize: "1.5rem",
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                  }}
                >
                  <img
                    src="/logo.png"
                    alt="OwnExperiences"
                    style={{ height: 40, marginRight: 10 }}
                    onError={(e) => (e.target.style.display = "none")}
                  />
                  OwnExperiences
                </Typography>
              </motion.div>
            </Toolbar>
          </Container>
        </AppBar>

        <Container
          component="main"
          maxWidth="lg"
          sx={{
            mt: 4,
            mb: 4,
            flex: 1,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Typography
              variant="h1"
              align="center"
              gutterBottom
              sx={{
                mb: 4,
                background: "linear-gradient(45deg, #2E3B55 30%, #FF6B6B 90%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Sentiment Analysis
            </Typography>
            <Typography
              variant="h4"
              align="center"
              color="text.secondary"
              sx={{ mb: 6 }}
            >
              Understand the emotions behind your text
            </Typography>
          </motion.div>
          <Paper sx={{ mb: 3 }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              centered
              sx={{
                "& .MuiTab-root": {
                  fontSize: "1.1rem",
                  py: 2,
                },
              }}
            >
              <Tab label="Single Analysis" />
              <Tab label="Bulk Analysis & Testing" />
            </Tabs>
          </Paper>
          {activeTab === 0 ? <SentimentAnalyzer /> : <BulkAnalyzer />}
        </Container>

        <Box
          component="footer"
          sx={{
            py: 3,
            px: 2,
            mt: "auto",
            backgroundColor: "background.paper",
            borderTop: "1px solid",
            borderColor: "divider",
          }}
        >
          <Container maxWidth="lg">
            <Typography variant="body2" color="text.secondary" align="center">
              © {new Date().getFullYear()} OwnExperiences.com | Empowering
              Decisions Through Sentiment Analysis
            </Typography>
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
