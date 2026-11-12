import axios from 'axios';
export const api = axios.create({baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api', timeout: 60000});
export type Review={review_id:number;review:string;hotel:string;ground_truth?:string;cluster?:number;similarity?:number;original_filename?:string};
export const errorText=(e:unknown)=>axios.isAxiosError(e)?(e.response?.data?.detail||e.message):'Something went wrong';
