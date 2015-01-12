// +build ignore

// This program generates Go wrappers for cuda sources.
// The cuda file should contain exactly one __global__ void.

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"github.com/mumax/3/util"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"text/scanner"
	"text/template"
)

func main() {
	flag.Parse()
	for _, fname := range flag.Args() {
		cuda2go(fname)
	}
}

// generate cuda wrapper for file.
func cuda2go(fname string) {
	// open cuda file
	f, err := os.Open(fname)
	util.PanicErr(err)
	defer f.Close()

	// read tokens
	var token []string
	var s scanner.Scanner
	s.Init(f)
	tok := s.Scan()
	for tok != scanner.EOF {
		if !filter(s.TokenText()) {
			token = append(token, s.TokenText())
		}
		tok = s.Scan()
	}

	// find function name and arguments
	funcname := ""
	argstart, argstop := -1, -1
	for i := 0; i < len(token); i++ {
		if token[i] == "__global__" {
			funcname = token[i+2]
			argstart = i + 4
		}
		if argstart > 0 && token[i] == ")" {
			argstop = i + 1
			break
		}
	}
	argl := token[argstart:argstop]

	// isolate individual arguments
	var args [][]string
	start := 0
	for i, a := range argl {
		if a == "," || a == ")" {
			args = append(args, argl[start:i])
			start = i + 1
		}
	}

	// separate arg names/types and make pointers Go-style
	argn := make([]string, len(args))
	argt := make([]string, len(args))
	for i := range args {
		if args[i][1] == "*" {
			args[i] = []string{args[i][0] + "*", args[i][2]}
		}
		argt[i] = typemap(args[i][0])
		argn[i] = args[i][1]
	}
	wrapgen(fname, funcname, argt, argn)
}

// translate C type to Go type.
func typemap(ctype string) string {
	if gotype, ok := tm[ctype]; ok {
		return gotype
	}
	panic(fmt.Errorf("unsupported cuda type: %v", ctype))
}

var tm = map[string]string{"float*": "unsafe.Pointer", "float": "float32", "int": "int", "uint8_t*": "unsafe.Pointer", "uint8_t": "byte"}

// template data
type Kernel struct {
	Name string
	ArgT []string
	ArgN []string
	PTX  map[int]string
}

var ls []string

// generate wrapper code from template
func wrapgen(filename, funcname string, argt, argn []string) {
	kernel := &Kernel{funcname, argt, argn, make(map[int]string)}

	// find corresponding .PTX files
	if ls == nil {
		dir, errd := os.Open(".")
		defer dir.Close()
		util.PanicErr(errd)
		var errls error
		ls, errls = dir.Readdirnames(-1)
		util.PanicErr(errls)
	}

	basename := util.NoExt(filename)
	for _, f := range ls {
		match, e := regexp.MatchString("^"+basename+"_*[0-9]..ptx", f)
		util.PanicErr(e)
		if match {
			cc, ei := strconv.Atoi(f[len(f)-len("00.ptx") : len(f)-len(".ptx")])
			util.PanicErr(ei)
			fmt.Println(basename, cc)
			kernel.PTX[cc] = filterptx(f)
		}
	}

	if len(kernel.PTX) == 0 {
		log.Fatal("no PTX files for ", filename)
	}

	wrapfname := basename + "_wrapper.go"
	wrapout, err := os.OpenFile(wrapfname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.PanicErr(err)
	defer wrapout.Close()
	util.PanicErr(templ.Execute(wrapout, kernel))
}

// wrapper code template text
const templText = `package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
)

// CUDA handle for {{.Name}} kernel
var {{.Name}}_code cu.Function

// Stores the arguments for {{.Name}} kernel invocation
type {{.Name}}_args_t struct{
	{{range $i, $_ := .ArgN}} arg_{{.}} {{index $.ArgT $i}}
	{{end}} argptr [{{len .ArgN}}]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for {{.Name}} kernel invocation
var {{.Name}}_args {{.Name}}_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	{{range $i, $t := .ArgN}} {{$.Name}}_args.argptr[{{$i}}] = unsafe.Pointer(&{{$.Name}}_args.arg_{{.}})
	{{end}} }

// Wrapper for {{.Name}} CUDA kernel, asynchronous.
func k_{{.Name}}_async ( {{range $i, $t := .ArgT}}{{index $.ArgN $i}} {{$t}}, {{end}} cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("{{.Name}}")
	}

	{{.Name}}_args.Lock()
	defer {{.Name}}_args.Unlock()

	if {{.Name}}_code == 0{
		{{.Name}}_code = fatbinLoad({{.Name}}_map, "{{.Name}}")
	}

	{{range $i, $t := .ArgN}} {{$.Name}}_args.arg_{{.}} = {{.}}
	{{end}}

	args := {{.Name}}_args.argptr[:]
	cu.LaunchKernel({{.Name}}_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("{{.Name}}")
	}
}

// maps compute capability on PTX code for {{.Name}} kernel.
var {{.Name}}_map = map[int]string{ 0: "" {{range $k, $v := .PTX}},
{{$k}}: {{$.Name}}_ptx_{{$k}} {{end}} }

// {{.Name}} PTX code for various compute capabilities.
const(
{{range $k, $v := .PTX}}  {{$.Name}}_ptx_{{$k}} = {{$v}}
 {{end}})
`

// wrapper code template
var templ = template.Must(template.New("wrap").Parse(templText))

// should token be filtered out of stream?
func filter(token string) bool {
	switch token {
	case "__restrict__":
		return true
	}
	return false
}

// Filter comments and ".file" entries from ptx code.
// They spoil the git history.
func filterptx(fname string) string {
	f, err := os.Open(fname)
	util.PanicErr(err)
	defer f.Close()
	in := bufio.NewReader(f)
	var out bytes.Buffer
	out.Write(([]byte)("`"))
	line, err := in.ReadBytes('\n')
	for err != io.EOF {
		util.PanicErr(err)
		if !bytes.HasPrefix(line, []byte("//")) && !bytes.HasPrefix(line, []byte("	.file")) {
			out.Write(line)
		}
		line, err = in.ReadBytes('\n')
	}
	out.Write(([]byte)("`"))
	return out.String()
}
