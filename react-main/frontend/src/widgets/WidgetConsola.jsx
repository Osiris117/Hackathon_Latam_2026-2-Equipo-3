import CardWidget from "../components/CardWidget";

export default function WidgetConsola(){

return(

<CardWidget titulo="Consola">

<textarea

style={{

width:"100%",

height:"180px",

background:"#000",

color:"#00FF00",

fontFamily:"monospace"

}}

readOnly

/>

</CardWidget>

);

}