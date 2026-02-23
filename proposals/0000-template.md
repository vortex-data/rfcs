# RFC 0000 - Template Mode

## Goals

1. Create RFCs
1. ??
1. Profit

## Diagrams

Here is an ASCII diagram:

```
     ╔ ═ ═ ═ ═ ═ ═ ═ ═ ═ ═                     
                          ║                    
     ║   BitPackedArray                        
                          ║                    
     ╚│═ ═ ═ ═ ═ ═ ═ ╤ ═ ═                     
      │              │                         
      │              │                         
      │              │                         
      │              │                         
      │              │                         
      │              │                         
      │              │  patch                  
      │              │  indices    ╔ ═ ═ ═ ═ ═ 
┌─────▼─────┐        ├─────────────▶ ArrayRef ║
│░░░░░░░░░░░│        │             ╚ ═ ═ ═ ═ ═ 
│░░Buffer░░░│        │                         
│░░░░░░░░░░░│        │  patch                  
└───────────┘        │  values     ╔ ═ ═ ═ ═ ═ 
   encoded           └─────────────▶ ArrayRef ║
                                   ╚ ═ ═ ═ ═ ═ 
```

We can have links, like https://github.com/vortex-data

BUT, we can also have [**LINKS**](https://vortex.dev) or [__links__](https://docs.vortex.dev)
