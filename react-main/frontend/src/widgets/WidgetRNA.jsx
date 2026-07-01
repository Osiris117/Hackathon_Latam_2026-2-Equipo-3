import { useState } from "react";

import {
Typography,
Button,
CircularProgress,
Stack
} from "@mui/material";

import CardWidget from "../components/CardWidget";
import StatusChip from "../components/StatusChip";

import { ejecutarRNA } from "../services/api";

export default function WidgetRNA(){

const[estado,setEstado]=useState("Activo");

const[loading,setLoading]=useState(false);

const[datos,setDatos]=useState({});

const entrenar=async()=>{

setLoading(true);

setEstado("Ejecutando");

try{

const r=await ejecutarRNA();

setDatos(r);

setEstado("Activo");

}
catch{

setEstado("Error");

}

setLoading(false);

};

return(

<CardWidget titulo="RNA">

<Stack spacing={2}>

<StatusChip estado={estado}/>

<Typography>

Accuracy

{datos.accuracy}

</Typography>

<Typography>

Arquitectura

{datos.arquitectura}

</Typography>

<Button

variant="contained"

onClick={entrenar}

>

{

loading?

<CircularProgress size={22}/>

:

"Entrenar"

}

</Button>

</Stack>

</CardWidget>

);

}