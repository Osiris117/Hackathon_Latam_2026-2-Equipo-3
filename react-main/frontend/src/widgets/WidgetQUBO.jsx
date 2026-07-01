import {Button,Typography,Stack} from "@mui/material";

import CardWidget from "../components/CardWidget";

export default function WidgetQUBO(){

return(

<CardWidget titulo="QUBO">

<Stack spacing={2}>

<Typography>

Variables binarias

</Typography>

<Typography>

Restricciones

</Typography>

<Button

variant="contained"

>

Construir

</Button>

</Stack>

</CardWidget>

);

}