import React from "react";
import ReactDOM from "react-dom/client";

import { CssBaseline } from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";

import "./index.css";
import App from "./App";

const theme = createTheme({
    palette: {
        mode: "light",
        primary: {
            main: "#1f6feb"
        },
        secondary: {
            main: "#168a49"
        },
        background: {
            default: "#eef1f6",
            paper: "#ffffff"
        }
    },
    shape: {
        borderRadius: 8
    }
});

ReactDOM.createRoot(document.getElementById("root")).render(

    <React.StrictMode>

        <ThemeProvider theme={theme}>

            <CssBaseline/>

            <App/>

        </ThemeProvider>

    </React.StrictMode>

);
