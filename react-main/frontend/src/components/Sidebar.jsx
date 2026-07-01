import {
    Drawer,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText
} from "@mui/material";

import DashboardIcon from "@mui/icons-material/Dashboard";
import ScienceIcon from "@mui/icons-material/Science";
import PsychologyIcon from "@mui/icons-material/Psychology";
import TimelineIcon from "@mui/icons-material/Timeline";
import MemoryIcon from "@mui/icons-material/Memory";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";

const menu = [

    {
        texto:"Dashboard",
        icono:<DashboardIcon/>
    },

    {
        texto:"Genético",
        icono:<ScienceIcon/>
    },

    {
        texto:"RNA",
        icono:<PsychologyIcon/>
    },

    {
        texto:"Modelado",
        icono:<TimelineIcon/>
    },

    {
        texto:"QUBO",
        icono:<MemoryIcon/>
    },

    {
        texto:"Quantum",
        icono:<AutoGraphIcon/>
    }

];

export default function Sidebar(){

return(

<Drawer
    variant="permanent"
    sx={{

        width:230,

        "& .MuiDrawer-paper":{

            width:230,

            background:"#0D1B2A",

            color:"white"

        }

    }}
>

<List>

{

menu.map((item,index)=>(

<ListItemButton key={index}>

<ListItemIcon
sx={{
color:"white"
}}
>

{item.icono}

</ListItemIcon>

<ListItemText
primary={item.texto}
/>

</ListItemButton>

))

}

</List>

</Drawer>

);

}