import { useState } from "react";

import {
Typography,
Button,
Stack,
CircularProgress
} from "@mui/material";

import CardWidget from "../components/CardWidget";

import StatusChip from "../components/StatusChip";

import { ejecutarModelado } from "../services/api";

export default function WidgetModelado(){

const[estado,setEstado]=useState("Activo");

const[loading,setLoading]=useState(false);

const[datos,setDatos]=useState({});

const ejecutar=async()=>{

setLoading(true);

setEstado("Ejecutando");

try{

const r=await ejecutarModelado();

setDatos(r);

setEstado("Activo");

}
catch{

setEstado("Error");

}

setLoading(false);

};

return(

<CardWidget titulo="Modelado">

<Stack spacing={2}>

<StatusChip estado={estado}/>

<Typography>

Semana

{datos.semana}

</Typography>

<Typography>

Nivel

{datos.nivel}

</Typography>

<Button

variant="contained"

onClick={ejecutar}

>

{

loading?

<CircularProgress size={22}/>

:

"Simular"

}

</Button>

</Stack>

</CardWidget>

);

}