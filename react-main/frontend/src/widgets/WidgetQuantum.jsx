import {Button,Typography,Stack} from "@mui/material";

import CardWidget from "../components/CardWidget";

export default function WidgetQuantum(){

return(

<CardWidget titulo="Computación Cuántica">

<Stack spacing={2}>

<Typography>

Backend

</Typography>

<Typography>

Estado

</Typography>

<Button

variant="contained"

>

Resolver

</Button>

</Stack>

</CardWidget>

);

}