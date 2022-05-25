#!/usr/bin/env node
const process = require('process')

const results = new Map()

function sum_words(chunk){
     
    chunk
        .toString()
        .split('\n')
        .forEach(entry =>{
            const [word, wc] = entry.split(' ')
            results.set(word, ~~wc+(results.get()|0))
        })
}

function finished(){
    for(const [word, wc] of results.entries()){
        console.log(`${word} ${wc}`)
    }
    process.exit(0)
}

process.stdin
    .on('data', sum_words)
    .on('end', finished)