#!/usr/bin/env node
const process = require('process')

const regExp = /[ \t\n\r]+/;
const white_chars = [' ', '\n', '\r', '\t'].map(c=>c.charCodeAt(0))
function is_sep(ch){return white_chars.includes(ch)}
let last = ''
const results = new Map()
function count_regexp(chunk){
    const str = last+chunk.toString()
    const spl = str.split(regExp).filter(w=>w!='')
    last = is_sep(chunk[chunk.length-1]) ? '' : spl.pop()
    spl.forEach(word => results.set(word, 1+(results.get(word)|0)))
}

function show_result(){
    for(const [word, wc] of results.entries()){
        console.log(`${word} ${wc}`)
    }
    process.exit(0)

}

process.stdin
    .on('data', count_regexp)
    .on('end', show_result)