import {
    Card,
    CardContent,
    Typography
} from "@mui/material";

export default function CardWidget({

    titulo,

    children

}){

return(

<Card
sx={

{

height:"100%",

background:"#16213E",

color:"white"

}

}
>

<CardContent>

<Typography
variant="h6"
sx={{
mb:2
}}
>

{titulo}

</Typography>

{children}

</CardContent>

</Card>

);

}