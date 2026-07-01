import Chip from "@mui/material/Chip";

export default function StatusChip({

    estado

}){

let color="default";

if(estado==="Activo") color="success";

if(estado==="Ejecutando") color="warning";

if(estado==="Error") color="error";

return(

<Chip

label={estado}

color={color}

/>

);

}