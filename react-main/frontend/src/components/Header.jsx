import { AppBar, Toolbar, Typography, Chip } from "@mui/material";

export default function Header() {

    return (

        <AppBar
            position="static"
            sx={{
                background: "#1B263B",
                boxShadow: 2
            }}
        >

            <Toolbar>

                <Typography
                    variant="h5"
                    sx={{
                        flexGrow: 1,
                        fontWeight: "bold"
                    }}
                >
                    Falcon Reservoir Dashboard
                </Typography>

                <Chip
                    label="Datos locales"
                    color="success"
                />

            </Toolbar>

        </AppBar>

    );

}
