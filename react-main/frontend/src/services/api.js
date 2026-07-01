import axios from "axios";

//======================================================
// CONFIGURACIÓN
//======================================================

const api = axios.create({

    baseURL: "http://localhost:8000",

    timeout: 300000

});

//======================================================
// ESTADO DEL SERVIDOR
//======================================================

export const obtenerEstado = async () => {

    const respuesta = await api.get("/estado");

    return respuesta.data;

};

//======================================================
// HOME
//======================================================

export const obtenerHome = async () => {

    const respuesta = await api.get("/");

    return respuesta.data;

};

//======================================================
// ALGORITMO GENÉTICO
//======================================================

export const ejecutarGenetico = async () => {

    const respuesta = await api.get("/genetico_v3");

    return respuesta.data;

};

//======================================================
// RNA
//======================================================

export const ejecutarRNA = async () => {

    const respuesta = await api.get("/rna");

    return respuesta.data;

};

//======================================================
// MODELADO
//======================================================

export const ejecutarModelado = async () => {

    const respuesta = await api.get("/modelado");

    return respuesta.data;

};

//======================================================
// QUBO
//======================================================

export const ejecutarQUBO = async () => {

    const respuesta = await api.get("/qubo");

    return respuesta.data;

};

//======================================================
// COMPUTACIÓN CUÁNTICA
//======================================================

export const ejecutarQuantum = async () => {

    const respuesta = await api.get("/quantum");

    return respuesta.data;

};

//======================================================
// EJECUTAR TODO
//======================================================

export const ejecutarTodo = async () => {

    const genetico = await ejecutarGenetico();

    const rna = await ejecutarRNA();

    const modelado = await ejecutarModelado();

    const qubo = await ejecutarQUBO();

    const quantum = await ejecutarQuantum();

    return {

        genetico,

        rna,

        modelado,

        qubo,

        quantum

    };

};

export default api;