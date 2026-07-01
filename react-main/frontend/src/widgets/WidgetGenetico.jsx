import { useState } from "react";

import {
    Typography,
    Button,
    CircularProgress,
    Stack
} from "@mui/material";

import CardWidget from "../components/CardWidget";
import StatusChip from "../components/StatusChip";

import { ejecutarGenetico } from "../services/api";

export default function WidgetGenetico(){

    const [estado,setEstado]=useState("Activo");
    const [loading,setLoading]=useState(false);
    const [datos,setDatos]=useState({});

    const ejecutar=async()=>{

        setLoading(true);

        setEstado("Ejecutando");

        try{

            const r=await ejecutarGenetico();

            setDatos(r);

            setEstado("Activo");

        }
        catch{

            setEstado("Error");

        }

        setLoading(false);

    };

    return(

<CardWidget titulo="Algoritmo Genético">

<Stack spacing={2}>

<StatusChip estado={estado}/>

<Typography>

Fitness

{datos.fitness}

</Typography>

<Typography>

Generaciones

{datos.generaciones}

</Typography>

<Button

variant="contained"

onClick={ejecutar}

disabled={loading}

>

{

loading?

<CircularProgress size={22}/>

:

"Ejecutar"

}

</Button>

</Stack>

</CardWidget>

);

}