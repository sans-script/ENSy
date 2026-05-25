import { SemanticSpace } from '../types';

export const semanticSpaceData: SemanticSpace = {
  "meta": {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "embedding_dim": 896,
    "total_tokens": 74,
    "multi_token_terms": [
      "machine learning",
      "artificial intelligence",
      "natural language",
      "data science",
      "computer vision",
      "cloud computing",
      "operating system",
      "pineapple",
      "lemon",
      "strawberry",
      "mango",
      "grape",
      "cherry",
      "peach",
      "plum",
      "green apple",
      "red apple",
      "watermelon",
      "blueberry",
      "raspberry",
      "blackberry",
      "kiwi",
      "avocado",
      "papaya",
      "coconut",
      "apricot",
      "harvest",
      "train model",
      "generate text",
      "load file",
      "export json",
      "analyze data",
      "scan document",
      "render scene",
      "clean cache"
    ],
    "umap_params": {
      "n_neighbors": 4,
      "min_dist": 0.5,
      "metric": "cosine"
    }
  },
  "nodes": [
    {
      "id": 0,
      "label": "document",
      "token_id": 6062,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 9.547721,
        "y2d": 7.391323,
        "x3d": 5.423237,
        "y3d": 2.238469,
        "z3d": 6.157989
      },
      "top5_neighbors": [
        {
          "label": "scan document",
          "cosine_similarity": 0.645582
        },
        {
          "label": "database",
          "cosine_similarity": 0.250779
        },
        {
          "label": "pdf",
          "cosine_similarity": 0.222168
        },
        {
          "label": "application",
          "cosine_similarity": 0.190225
        },
        {
          "label": "load file",
          "cosine_similarity": 0.176827
        }
      ]
    },
    {
      "id": 1,
      "label": "file",
      "token_id": 1192,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 8.042846,
        "y2d": 7.082941,
        "x3d": 6.253492,
        "y3d": 1.403474,
        "z3d": 5.535983
      },
      "top5_neighbors": [
        {
          "label": "load file",
          "cosine_similarity": 0.608092
        },
        {
          "label": "data",
          "cosine_similarity": 0.200268
        },
        {
          "label": "data science",
          "cosine_similarity": 0.189556
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.181519
        },
        {
          "label": "code",
          "cosine_similarity": 0.179182
        }
      ]
    },
    {
      "id": 2,
      "label": "pdf",
      "token_id": 11828,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 9.405147,
        "y2d": 7.363185,
        "x3d": 5.058619,
        "y3d": 2.043155,
        "z3d": 5.92064
      },
      "top5_neighbors": [
        {
          "label": "document",
          "cosine_similarity": 0.222168
        },
        {
          "label": "scan document",
          "cosine_similarity": 0.209714
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.198211
        },
        {
          "label": "save",
          "cosine_similarity": 0.172537
        },
        {
          "label": "banana",
          "cosine_similarity": 0.171299
        }
      ]
    },
    {
      "id": 3,
      "label": "text",
      "token_id": 1318,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 7.145878,
        "y2d": 7.460289,
        "x3d": 7.122631,
        "y3d": 0.690413,
        "z3d": 4.689847
      },
      "top5_neighbors": [
        {
          "label": "generate text",
          "cosine_similarity": 0.613269
        },
        {
          "label": "data",
          "cosine_similarity": 0.219866
        },
        {
          "label": "data science",
          "cosine_similarity": 0.177025
        },
        {
          "label": "file",
          "cosine_similarity": 0.169146
        },
        {
          "label": "code",
          "cosine_similarity": 0.160064
        }
      ]
    },
    {
      "id": 4,
      "label": "data",
      "token_id": 691,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 7.688931,
        "y2d": 7.806448,
        "x3d": 7.235157,
        "y3d": 1.197097,
        "z3d": 5.025476
      },
      "top5_neighbors": [
        {
          "label": "data science",
          "cosine_similarity": 0.763873
        },
        {
          "label": "analyze data",
          "cosine_similarity": 0.580871
        },
        {
          "label": "text",
          "cosine_similarity": 0.219866
        },
        {
          "label": "database",
          "cosine_similarity": 0.218068
        },
        {
          "label": "file",
          "cosine_similarity": 0.200268
        }
      ]
    },
    {
      "id": 5,
      "label": "code",
      "token_id": 1851,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 12.504274,
        "y2d": 7.341824,
        "x3d": 6.650489,
        "y3d": -0.880428,
        "z3d": 6.859124
      },
      "top5_neighbors": [
        {
          "label": "encode",
          "cosine_similarity": 0.238921
        },
        {
          "label": "computer",
          "cosine_similarity": 0.208699
        },
        {
          "label": "data",
          "cosine_similarity": 0.192101
        },
        {
          "label": "write",
          "cosine_similarity": 0.181383
        },
        {
          "label": "file",
          "cosine_similarity": 0.179182
        }
      ]
    },
    {
      "id": 6,
      "label": "computer",
      "token_id": 43111,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 11.622568,
        "y2d": 6.124675,
        "x3d": 6.945387,
        "y3d": -0.266569,
        "z3d": 8.001542
      },
      "top5_neighbors": [
        {
          "label": "computer vision",
          "cosine_similarity": 0.742962
        },
        {
          "label": "cloud computing",
          "cosine_similarity": 0.437718
        },
        {
          "label": "machine learning",
          "cosine_similarity": 0.28567
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.284549
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.279086
        }
      ]
    },
    {
      "id": 7,
      "label": "hardware",
      "token_id": 68215,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 11.331059,
        "y2d": 7.343619,
        "x3d": 7.069432,
        "y3d": 0.372323,
        "z3d": 7.145692
      },
      "top5_neighbors": [
        {
          "label": "encryption",
          "cosine_similarity": 0.317766
        },
        {
          "label": "database",
          "cosine_similarity": 0.309805
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.285878
        },
        {
          "label": "computer",
          "cosine_similarity": 0.284549
        },
        {
          "label": "algorithm",
          "cosine_similarity": 0.27871
        }
      ]
    },
    {
      "id": 8,
      "label": "byte",
      "token_id": 3782,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 14.495296,
        "y2d": 5.38207,
        "x3d": 6.17491,
        "y3d": -2.940223,
        "z3d": 8.728344
      },
      "top5_neighbors": [
        {
          "label": "watermelon",
          "cosine_similarity": 0.231902
        },
        {
          "label": "blueberry",
          "cosine_similarity": 0.228803
        },
        {
          "label": "binary",
          "cosine_similarity": 0.225237
        },
        {
          "label": "blackberry",
          "cosine_similarity": 0.222588
        },
        {
          "label": "computer",
          "cosine_similarity": 0.209658
        }
      ]
    },
    {
      "id": 9,
      "label": "format",
      "token_id": 2243,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 10.710885,
        "y2d": 8.35699,
        "x3d": 6.488948,
        "y3d": -0.380869,
        "z3d": 5.837117
      },
      "top5_neighbors": [
        {
          "label": "parse",
          "cosine_similarity": 0.226261
        },
        {
          "label": "generate text",
          "cosine_similarity": 0.188024
        },
        {
          "label": "algorithm",
          "cosine_similarity": 0.186464
        },
        {
          "label": "encode",
          "cosine_similarity": 0.170496
        },
        {
          "label": "write",
          "cosine_similarity": 0.168682
        }
      ]
    },
    {
      "id": 10,
      "label": "binary",
      "token_id": 25891,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 13.707213,
        "y2d": 7.414529,
        "x3d": 7.28922,
        "y3d": -1.965818,
        "z3d": 7.83687
      },
      "top5_neighbors": [
        {
          "label": "compression",
          "cosine_similarity": 0.253748
        },
        {
          "label": "encryption",
          "cosine_similarity": 0.242959
        },
        {
          "label": "byte",
          "cosine_similarity": 0.225237
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.221364
        },
        {
          "label": "blackberry",
          "cosine_similarity": 0.220498
        }
      ]
    },
    {
      "id": 11,
      "label": "memory",
      "token_id": 17269,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 12.946017,
        "y2d": 5.150375,
        "x3d": 6.172427,
        "y3d": -1.421364,
        "z3d": 8.809911
      },
      "top5_neighbors": [
        {
          "label": "watermelon",
          "cosine_similarity": 0.266131
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.257075
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.251437
        },
        {
          "label": "database",
          "cosine_similarity": 0.239422
        },
        {
          "label": "network",
          "cosine_similarity": 0.23445
        }
      ]
    },
    {
      "id": 12,
      "label": "machine learning",
      "token_id": 32056,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 12.063491,
        "y2d": 6.558045,
        "x3d": 7.638769,
        "y3d": 0.062986,
        "z3d": 8.250289
      },
      "top5_neighbors": [
        {
          "label": "computer",
          "cosine_similarity": 0.28567
        },
        {
          "label": "network",
          "cosine_similarity": 0.26562
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.257287
        },
        {
          "label": "algorithm",
          "cosine_similarity": 0.235122
        },
        {
          "label": "memory",
          "cosine_similarity": 0.233907
        }
      ]
    },
    {
      "id": 13,
      "label": "artificial intelligence",
      "token_id": 471,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 11.916364,
        "y2d": 8.011972,
        "x3d": 7.561344,
        "y3d": -0.065864,
        "z3d": 7.120827
      },
      "top5_neighbors": [
        {
          "label": "algorithm",
          "cosine_similarity": 0.240651
        },
        {
          "label": "machine learning",
          "cosine_similarity": 0.226808
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.211909
        },
        {
          "label": "computer",
          "cosine_similarity": 0.208402
        },
        {
          "label": "cloud computing",
          "cosine_similarity": 0.208199
        }
      ]
    },
    {
      "id": 14,
      "label": "natural language",
      "token_id": 52880,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 8.798536,
        "y2d": 8.300261,
        "x3d": 7.670311,
        "y3d": 0.539803,
        "z3d": 6.041625
      },
      "top5_neighbors": [
        {
          "label": "analyze data",
          "cosine_similarity": 0.216433
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.213055
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.209929
        },
        {
          "label": "computer",
          "cosine_similarity": 0.208112
        },
        {
          "label": "machine learning",
          "cosine_similarity": 0.205927
        }
      ]
    },
    {
      "id": 15,
      "label": "data science",
      "token_id": 691,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 7.555509,
        "y2d": 8.097966,
        "x3d": 7.044821,
        "y3d": 1.49946,
        "z3d": 5.447984
      },
      "top5_neighbors": [
        {
          "label": "data",
          "cosine_similarity": 0.763873
        },
        {
          "label": "analyze data",
          "cosine_similarity": 0.50435
        },
        {
          "label": "database",
          "cosine_similarity": 0.191585
        },
        {
          "label": "file",
          "cosine_similarity": 0.189556
        },
        {
          "label": "text",
          "cosine_similarity": 0.177025
        }
      ]
    },
    {
      "id": 16,
      "label": "computer vision",
      "token_id": 43111,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 11.892428,
        "y2d": 5.458441,
        "x3d": 6.859045,
        "y3d": -0.516856,
        "z3d": 8.568256
      },
      "top5_neighbors": [
        {
          "label": "computer",
          "cosine_similarity": 0.742962
        },
        {
          "label": "cloud computing",
          "cosine_similarity": 0.348375
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.285878
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.276675
        },
        {
          "label": "compression",
          "cosine_similarity": 0.272319
        }
      ]
    },
    {
      "id": 17,
      "label": "cloud computing",
      "token_id": 12361,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 11.249672,
        "y2d": 5.419665,
        "x3d": 6.248465,
        "y3d": -0.172624,
        "z3d": 8.204825
      },
      "top5_neighbors": [
        {
          "label": "computer",
          "cosine_similarity": 0.437718
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.348375
        },
        {
          "label": "encryption",
          "cosine_similarity": 0.273122
        },
        {
          "label": "clean cache",
          "cosine_similarity": 0.231408
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.225276
        }
      ]
    },
    {
      "id": 18,
      "label": "operating system",
      "token_id": 3296,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 11.494541,
        "y2d": 7.116228,
        "x3d": 7.630167,
        "y3d": 0.524224,
        "z3d": 7.441086
      },
      "top5_neighbors": [
        {
          "label": "hardware",
          "cosine_similarity": 0.166708
        },
        {
          "label": "network",
          "cosine_similarity": 0.16262
        },
        {
          "label": "computer",
          "cosine_similarity": 0.154801
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.153922
        },
        {
          "label": "algorithm",
          "cosine_similarity": 0.147582
        }
      ]
    },
    {
      "id": 19,
      "label": "database",
      "token_id": 12216,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 10.35865,
        "y2d": 7.304259,
        "x3d": 6.765138,
        "y3d": 1.282739,
        "z3d": 6.790631
      },
      "top5_neighbors": [
        {
          "label": "network",
          "cosine_similarity": 0.314031
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.309805
        },
        {
          "label": "server",
          "cosine_similarity": 0.278221
        },
        {
          "label": "algorithm",
          "cosine_similarity": 0.264123
        },
        {
          "label": "computer",
          "cosine_similarity": 0.258061
        }
      ]
    },
    {
      "id": 20,
      "label": "network",
      "token_id": 17511,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 10.934344,
        "y2d": 6.927495,
        "x3d": 7.45581,
        "y3d": 0.865089,
        "z3d": 7.483086
      },
      "top5_neighbors": [
        {
          "label": "database",
          "cosine_similarity": 0.314031
        },
        {
          "label": "machine learning",
          "cosine_similarity": 0.26562
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.242635
        },
        {
          "label": "memory",
          "cosine_similarity": 0.23445
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.219096
        }
      ]
    },
    {
      "id": 21,
      "label": "server",
      "token_id": 4030,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 10.409919,
        "y2d": 6.176381,
        "x3d": 6.216903,
        "y3d": 1.197834,
        "z3d": 7.33668
      },
      "top5_neighbors": [
        {
          "label": "database",
          "cosine_similarity": 0.278221
        },
        {
          "label": "computer",
          "cosine_similarity": 0.215045
        },
        {
          "label": "store",
          "cosine_similarity": 0.211661
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.20689
        },
        {
          "label": "delete",
          "cosine_similarity": 0.196739
        }
      ]
    },
    {
      "id": 22,
      "label": "application",
      "token_id": 5132,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 15.436787,
        "y2d": 7.921062,
        "x3d": 5.762375,
        "y3d": -2.585546,
        "z3d": 6.004567
      },
      "top5_neighbors": [
        {
          "label": "pineapple",
          "cosine_similarity": 0.234042
        },
        {
          "label": "algorithm",
          "cosine_similarity": 0.221241
        },
        {
          "label": "apple",
          "cosine_similarity": 0.21392
        },
        {
          "label": "database",
          "cosine_similarity": 0.204809
        },
        {
          "label": "compression",
          "cosine_similarity": 0.191341
        }
      ]
    },
    {
      "id": 23,
      "label": "algorithm",
      "token_id": 19688,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 11.346721,
        "y2d": 8.31402,
        "x3d": 6.94001,
        "y3d": -0.029258,
        "z3d": 6.662471
      },
      "top5_neighbors": [
        {
          "label": "hardware",
          "cosine_similarity": 0.27871
        },
        {
          "label": "database",
          "cosine_similarity": 0.264123
        },
        {
          "label": "computer",
          "cosine_similarity": 0.257787
        },
        {
          "label": "encryption",
          "cosine_similarity": 0.243312
        },
        {
          "label": "artificial intelligence",
          "cosine_similarity": 0.240651
        }
      ]
    },
    {
      "id": 24,
      "label": "encryption",
      "token_id": 79937,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 13.123037,
        "y2d": 8.098102,
        "x3d": 7.107915,
        "y3d": -1.572899,
        "z3d": 7.133038
      },
      "top5_neighbors": [
        {
          "label": "compression",
          "cosine_similarity": 0.432034
        },
        {
          "label": "encode",
          "cosine_similarity": 0.36403
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.317766
        },
        {
          "label": "cloud computing",
          "cosine_similarity": 0.273122
        },
        {
          "label": "banana",
          "cosine_similarity": 0.259836
        }
      ]
    },
    {
      "id": 25,
      "label": "compression",
      "token_id": 83192,
      "cluster": "Technology/File",
      "color": "#378ADD",
      "coords": {
        "x2d": 13.329578,
        "y2d": 7.540361,
        "x3d": 7.238979,
        "y3d": -1.44701,
        "z3d": 7.621518
      },
      "top5_neighbors": [
        {
          "label": "encryption",
          "cosine_similarity": 0.432034
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.272319
        },
        {
          "label": "computer",
          "cosine_similarity": 0.265503
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.262982
        },
        {
          "label": "database",
          "cosine_similarity": 0.257186
        }
      ]
    },
    {
      "id": 26,
      "label": "apple",
      "token_id": 22377,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 16.13143,
        "y2d": 7.022858,
        "x3d": 5.390886,
        "y3d": -3.376008,
        "z3d": 6.601261
      },
      "top5_neighbors": [
        {
          "label": "pineapple",
          "cosine_similarity": 0.746344
        },
        {
          "label": "green apple",
          "cosine_similarity": 0.516776
        },
        {
          "label": "red apple",
          "cosine_similarity": 0.40449
        },
        {
          "label": "banana",
          "cosine_similarity": 0.37185
        },
        {
          "label": "fruit",
          "cosine_similarity": 0.300196
        }
      ]
    },
    {
      "id": 27,
      "label": "pineapple",
      "token_id": 38038,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.669086,
        "y2d": 7.40716,
        "x3d": 5.800243,
        "y3d": -3.220485,
        "z3d": 6.643961
      },
      "top5_neighbors": [
        {
          "label": "apple",
          "cosine_similarity": 0.746344
        },
        {
          "label": "green apple",
          "cosine_similarity": 0.461171
        },
        {
          "label": "banana",
          "cosine_similarity": 0.39628
        },
        {
          "label": "red apple",
          "cosine_similarity": 0.341992
        },
        {
          "label": "fruit",
          "cosine_similarity": 0.326329
        }
      ]
    },
    {
      "id": 28,
      "label": "banana",
      "token_id": 87747,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 16.406971,
        "y2d": 7.223103,
        "x3d": 5.482299,
        "y3d": -2.805247,
        "z3d": 6.729419
      },
      "top5_neighbors": [
        {
          "label": "pineapple",
          "cosine_similarity": 0.39628
        },
        {
          "label": "apple",
          "cosine_similarity": 0.37185
        },
        {
          "label": "green apple",
          "cosine_similarity": 0.311188
        },
        {
          "label": "fruit",
          "cosine_similarity": 0.271132
        },
        {
          "label": "computer",
          "cosine_similarity": 0.264816
        }
      ]
    },
    {
      "id": 29,
      "label": "fruit",
      "token_id": 35598,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.028821,
        "y2d": 6.288575,
        "x3d": 5.981028,
        "y3d": -3.652001,
        "z3d": 7.049294
      },
      "top5_neighbors": [
        {
          "label": "green apple",
          "cosine_similarity": 0.329522
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.326329
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.315162
        },
        {
          "label": "apple",
          "cosine_similarity": 0.300196
        },
        {
          "label": "banana",
          "cosine_similarity": 0.271132
        }
      ]
    },
    {
      "id": 30,
      "label": "orange",
      "token_id": 34164,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 16.641378,
        "y2d": 5.725869,
        "x3d": 5.136411,
        "y3d": -4.511817,
        "z3d": 6.647924
      },
      "top5_neighbors": [
        {
          "label": "green apple",
          "cosine_similarity": 0.359454
        },
        {
          "label": "blueberry",
          "cosine_similarity": 0.288833
        },
        {
          "label": "red apple",
          "cosine_similarity": 0.273609
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.259568
        },
        {
          "label": "blackberry",
          "cosine_similarity": 0.250749
        }
      ]
    },
    {
      "id": 31,
      "label": "lemon",
      "token_id": 273,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.052771,
        "y2d": 2.255188,
        "x3d": 5.072327,
        "y3d": -4.176151,
        "z3d": 11.573121
      },
      "top5_neighbors": [
        {
          "label": "peach",
          "cosine_similarity": 0.252254
        },
        {
          "label": "plum",
          "cosine_similarity": 0.168846
        },
        {
          "label": "apricot",
          "cosine_similarity": 0.148794
        },
        {
          "label": "mango",
          "cosine_similarity": 0.126375
        },
        {
          "label": "strawberry",
          "cosine_similarity": 0.12637
        }
      ]
    },
    {
      "id": 32,
      "label": "strawberry",
      "token_id": 495,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.06503,
        "y2d": 4.070513,
        "x3d": 5.199049,
        "y3d": -3.805519,
        "z3d": 9.907175
      },
      "top5_neighbors": [
        {
          "label": "blueberry",
          "cosine_similarity": 0.537926
        },
        {
          "label": "blackberry",
          "cosine_similarity": 0.492737
        },
        {
          "label": "raspberry",
          "cosine_similarity": 0.464534
        },
        {
          "label": "grape",
          "cosine_similarity": 0.211345
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.204429
        }
      ]
    },
    {
      "id": 33,
      "label": "mango",
      "token_id": 76,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 16.495079,
        "y2d": 3.385129,
        "x3d": 6.037925,
        "y3d": -4.757537,
        "z3d": 10.893093
      },
      "top5_neighbors": [
        {
          "label": "papaya",
          "cosine_similarity": 0.273949
        },
        {
          "label": "raspberry",
          "cosine_similarity": 0.240506
        },
        {
          "label": "plum",
          "cosine_similarity": 0.154421
        },
        {
          "label": "peach",
          "cosine_similarity": 0.126992
        },
        {
          "label": "lemon",
          "cosine_similarity": 0.126375
        }
      ]
    },
    {
      "id": 34,
      "label": "grape",
      "token_id": 901,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 14.99415,
        "y2d": 3.199654,
        "x3d": 5.041304,
        "y3d": -3.481348,
        "z3d": 10.920374
      },
      "top5_neighbors": [
        {
          "label": "peach",
          "cosine_similarity": 0.239016
        },
        {
          "label": "apricot",
          "cosine_similarity": 0.217557
        },
        {
          "label": "strawberry",
          "cosine_similarity": 0.211345
        },
        {
          "label": "papaya",
          "cosine_similarity": 0.181066
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.179875
        }
      ]
    },
    {
      "id": 35,
      "label": "cherry",
      "token_id": 331,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 14.792229,
        "y2d": 3.009649,
        "x3d": 4.550222,
        "y3d": -3.982946,
        "z3d": 10.686862
      },
      "top5_neighbors": [
        {
          "label": "peach",
          "cosine_similarity": 0.200492
        },
        {
          "label": "strawberry",
          "cosine_similarity": 0.18695
        },
        {
          "label": "plum",
          "cosine_similarity": 0.156434
        },
        {
          "label": "raspberry",
          "cosine_similarity": 0.146879
        },
        {
          "label": "blueberry",
          "cosine_similarity": 0.143641
        }
      ]
    },
    {
      "id": 36,
      "label": "peach",
      "token_id": 375,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 14.581061,
        "y2d": 2.655913,
        "x3d": 4.707121,
        "y3d": -3.769639,
        "z3d": 11.379704
      },
      "top5_neighbors": [
        {
          "label": "lemon",
          "cosine_similarity": 0.252254
        },
        {
          "label": "grape",
          "cosine_similarity": 0.239016
        },
        {
          "label": "cherry",
          "cosine_similarity": 0.200492
        },
        {
          "label": "papaya",
          "cosine_similarity": 0.193588
        },
        {
          "label": "plum",
          "cosine_similarity": 0.175756
        }
      ]
    },
    {
      "id": 37,
      "label": "plum",
      "token_id": 500,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.754372,
        "y2d": 2.468897,
        "x3d": 5.179778,
        "y3d": -4.508122,
        "z3d": 11.405121
      },
      "top5_neighbors": [
        {
          "label": "papaya",
          "cosine_similarity": 0.228985
        },
        {
          "label": "apricot",
          "cosine_similarity": 0.188832
        },
        {
          "label": "peach",
          "cosine_similarity": 0.175756
        },
        {
          "label": "lemon",
          "cosine_similarity": 0.168846
        },
        {
          "label": "cherry",
          "cosine_similarity": 0.156434
        }
      ]
    },
    {
      "id": 38,
      "label": "green apple",
      "token_id": 13250,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 16.321421,
        "y2d": 6.360282,
        "x3d": 5.548813,
        "y3d": -4.066102,
        "z3d": 6.629272
      },
      "top5_neighbors": [
        {
          "label": "red apple",
          "cosine_similarity": 0.60828
        },
        {
          "label": "apple",
          "cosine_similarity": 0.516776
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.461171
        },
        {
          "label": "blueberry",
          "cosine_similarity": 0.395077
        },
        {
          "label": "orange",
          "cosine_similarity": 0.359454
        }
      ]
    },
    {
      "id": 39,
      "label": "red apple",
      "token_id": 1151,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 16.738024,
        "y2d": 6.536994,
        "x3d": 5.334659,
        "y3d": -4.026999,
        "z3d": 6.089931
      },
      "top5_neighbors": [
        {
          "label": "green apple",
          "cosine_similarity": 0.60828
        },
        {
          "label": "apple",
          "cosine_similarity": 0.40449
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.341992
        },
        {
          "label": "blueberry",
          "cosine_similarity": 0.276293
        },
        {
          "label": "orange",
          "cosine_similarity": 0.273609
        }
      ]
    },
    {
      "id": 40,
      "label": "watermelon",
      "token_id": 12987,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 14.114037,
        "y2d": 5.504415,
        "x3d": 5.757753,
        "y3d": -2.487471,
        "z3d": 8.568828
      },
      "top5_neighbors": [
        {
          "label": "blueberry",
          "cosine_similarity": 0.31886
        },
        {
          "label": "fruit",
          "cosine_similarity": 0.315162
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.303192
        },
        {
          "label": "blackberry",
          "cosine_similarity": 0.287101
        },
        {
          "label": "memory",
          "cosine_similarity": 0.266131
        }
      ]
    },
    {
      "id": 41,
      "label": "blueberry",
      "token_id": 12203,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.378168,
        "y2d": 4.863517,
        "x3d": 5.670627,
        "y3d": -3.787776,
        "z3d": 9.050323
      },
      "top5_neighbors": [
        {
          "label": "blackberry",
          "cosine_similarity": 0.764371
        },
        {
          "label": "strawberry",
          "cosine_similarity": 0.537926
        },
        {
          "label": "raspberry",
          "cosine_similarity": 0.497209
        },
        {
          "label": "green apple",
          "cosine_similarity": 0.395077
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.31886
        }
      ]
    },
    {
      "id": 42,
      "label": "raspberry",
      "token_id": 12784,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.952119,
        "y2d": 3.853729,
        "x3d": 5.868187,
        "y3d": -4.388755,
        "z3d": 10.110435
      },
      "top5_neighbors": [
        {
          "label": "blueberry",
          "cosine_similarity": 0.497209
        },
        {
          "label": "blackberry",
          "cosine_similarity": 0.487278
        },
        {
          "label": "strawberry",
          "cosine_similarity": 0.464534
        },
        {
          "label": "papaya",
          "cosine_similarity": 0.414119
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.249176
        }
      ]
    },
    {
      "id": 43,
      "label": "blackberry",
      "token_id": 11453,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.551661,
        "y2d": 4.303609,
        "x3d": 5.587132,
        "y3d": -4.162944,
        "z3d": 9.576979
      },
      "top5_neighbors": [
        {
          "label": "blueberry",
          "cosine_similarity": 0.764371
        },
        {
          "label": "strawberry",
          "cosine_similarity": 0.492737
        },
        {
          "label": "raspberry",
          "cosine_similarity": 0.487278
        },
        {
          "label": "green apple",
          "cosine_similarity": 0.30497
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.287101
        }
      ]
    },
    {
      "id": 44,
      "label": "kiwi",
      "token_id": 6642,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.617461,
        "y2d": 6.039171,
        "x3d": 5.630569,
        "y3d": -3.68804,
        "z3d": 7.815737
      },
      "top5_neighbors": [
        {
          "label": "pineapple",
          "cosine_similarity": 0.191797
        },
        {
          "label": "blueberry",
          "cosine_similarity": 0.188206
        },
        {
          "label": "apple",
          "cosine_similarity": 0.171779
        },
        {
          "label": "strawberry",
          "cosine_similarity": 0.166248
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.161322
        }
      ]
    },
    {
      "id": 45,
      "label": "avocado",
      "token_id": 402,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 14.565716,
        "y2d": 3.979727,
        "x3d": 5.134645,
        "y3d": -3.190993,
        "z3d": 9.911543
      },
      "top5_neighbors": [
        {
          "label": "strawberry",
          "cosine_similarity": 0.189736
        },
        {
          "label": "grape",
          "cosine_similarity": 0.177545
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.17627
        },
        {
          "label": "apricot",
          "cosine_similarity": 0.169251
        },
        {
          "label": "coconut",
          "cosine_similarity": 0.167693
        }
      ]
    },
    {
      "id": 46,
      "label": "papaya",
      "token_id": 79,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 16.371088,
        "y2d": 3.039306,
        "x3d": 5.687871,
        "y3d": -4.680123,
        "z3d": 11.087264
      },
      "top5_neighbors": [
        {
          "label": "raspberry",
          "cosine_similarity": 0.414119
        },
        {
          "label": "apricot",
          "cosine_similarity": 0.408068
        },
        {
          "label": "mango",
          "cosine_similarity": 0.273949
        },
        {
          "label": "plum",
          "cosine_similarity": 0.228985
        },
        {
          "label": "peach",
          "cosine_similarity": 0.193588
        }
      ]
    },
    {
      "id": 47,
      "label": "coconut",
      "token_id": 1015,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.737643,
        "y2d": 7.147069,
        "x3d": 5.016679,
        "y3d": -3.217216,
        "z3d": 6.233254
      },
      "top5_neighbors": [
        {
          "label": "pineapple",
          "cosine_similarity": 0.184793
        },
        {
          "label": "pear",
          "cosine_similarity": 0.17612
        },
        {
          "label": "apple",
          "cosine_similarity": 0.175111
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.174362
        },
        {
          "label": "avocado",
          "cosine_similarity": 0.167693
        }
      ]
    },
    {
      "id": 48,
      "label": "pear",
      "token_id": 8015,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.398167,
        "y2d": 6.903389,
        "x3d": 4.953346,
        "y3d": -3.281113,
        "z3d": 7.161874
      },
      "top5_neighbors": [
        {
          "label": "pineapple",
          "cosine_similarity": 0.276481
        },
        {
          "label": "apple",
          "cosine_similarity": 0.241236
        },
        {
          "label": "green apple",
          "cosine_similarity": 0.238969
        },
        {
          "label": "blueberry",
          "cosine_similarity": 0.226998
        },
        {
          "label": "banana",
          "cosine_similarity": 0.224848
        }
      ]
    },
    {
      "id": 49,
      "label": "apricot",
      "token_id": 391,
      "cluster": "Biological/Fruit",
      "color": "#1D9E75",
      "coords": {
        "x2d": 15.840424,
        "y2d": 2.859565,
        "x3d": 5.631757,
        "y3d": -4.112402,
        "z3d": 10.995645
      },
      "top5_neighbors": [
        {
          "label": "papaya",
          "cosine_similarity": 0.408068
        },
        {
          "label": "grape",
          "cosine_similarity": 0.217557
        },
        {
          "label": "raspberry",
          "cosine_similarity": 0.214776
        },
        {
          "label": "plum",
          "cosine_similarity": 0.188832
        },
        {
          "label": "peach",
          "cosine_similarity": 0.171814
        }
      ]
    },
    {
      "id": 50,
      "label": "write",
      "token_id": 4934,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 9.347237,
        "y2d": 4.916259,
        "x3d": 4.696916,
        "y3d": 0.292685,
        "z3d": 7.346329
      },
      "top5_neighbors": [
        {
          "label": "read",
          "cosine_similarity": 0.40815
        },
        {
          "label": "save",
          "cosine_similarity": 0.280659
        },
        {
          "label": "create",
          "cosine_similarity": 0.269559
        },
        {
          "label": "encode",
          "cosine_similarity": 0.266623
        },
        {
          "label": "delete",
          "cosine_similarity": 0.236487
        }
      ]
    },
    {
      "id": 51,
      "label": "read",
      "token_id": 878,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 8.860572,
        "y2d": 5.573953,
        "x3d": 4.928474,
        "y3d": 0.074589,
        "z3d": 6.503253
      },
      "top5_neighbors": [
        {
          "label": "write",
          "cosine_similarity": 0.40815
        },
        {
          "label": "parse",
          "cosine_similarity": 0.215992
        },
        {
          "label": "load file",
          "cosine_similarity": 0.193642
        },
        {
          "label": "open",
          "cosine_similarity": 0.187916
        },
        {
          "label": "red apple",
          "cosine_similarity": 0.183872
        }
      ]
    },
    {
      "id": 52,
      "label": "open",
      "token_id": 2508,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 8.442968,
        "y2d": 4.966852,
        "x3d": 4.132061,
        "y3d": 0.270789,
        "z3d": 6.642662
      },
      "top5_neighbors": [
        {
          "label": "create",
          "cosine_similarity": 0.201594
        },
        {
          "label": "read",
          "cosine_similarity": 0.187916
        },
        {
          "label": "apple",
          "cosine_similarity": 0.169281
        },
        {
          "label": "update",
          "cosine_similarity": 0.156138
        },
        {
          "label": "load file",
          "cosine_similarity": 0.153625
        }
      ]
    },
    {
      "id": 53,
      "label": "save",
      "token_id": 6628,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 10.070111,
        "y2d": 4.472283,
        "x3d": 4.949653,
        "y3d": 0.543775,
        "z3d": 8.255862
      },
      "top5_neighbors": [
        {
          "label": "delete",
          "cosine_similarity": 0.293336
        },
        {
          "label": "write",
          "cosine_similarity": 0.280659
        },
        {
          "label": "clean cache",
          "cosine_similarity": 0.214529
        },
        {
          "label": "render scene",
          "cosine_similarity": 0.213007
        },
        {
          "label": "store",
          "cosine_similarity": 0.206193
        }
      ]
    },
    {
      "id": 54,
      "label": "parse",
      "token_id": 6400,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 11.213348,
        "y2d": 8.093912,
        "x3d": 5.982708,
        "y3d": -0.940732,
        "z3d": 6.354324
      },
      "top5_neighbors": [
        {
          "label": "encode",
          "cosine_similarity": 0.227304
        },
        {
          "label": "format",
          "cosine_similarity": 0.226261
        },
        {
          "label": "read",
          "cosine_similarity": 0.215992
        },
        {
          "label": "eat",
          "cosine_similarity": 0.189642
        },
        {
          "label": "write",
          "cosine_similarity": 0.189121
        }
      ]
    },
    {
      "id": 55,
      "label": "encode",
      "token_id": 6180,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 12.646619,
        "y2d": 7.764834,
        "x3d": 6.265872,
        "y3d": -1.357784,
        "z3d": 6.952815
      },
      "top5_neighbors": [
        {
          "label": "encryption",
          "cosine_similarity": 0.36403
        },
        {
          "label": "write",
          "cosine_similarity": 0.266623
        },
        {
          "label": "compression",
          "cosine_similarity": 0.251032
        },
        {
          "label": "code",
          "cosine_similarity": 0.238921
        },
        {
          "label": "eat",
          "cosine_similarity": 0.227456
        }
      ]
    },
    {
      "id": 56,
      "label": "store",
      "token_id": 4314,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 10.012746,
        "y2d": 5.356052,
        "x3d": 5.348237,
        "y3d": 1.068983,
        "z3d": 7.569449
      },
      "top5_neighbors": [
        {
          "label": "server",
          "cosine_similarity": 0.211661
        },
        {
          "label": "save",
          "cosine_similarity": 0.206193
        },
        {
          "label": "write",
          "cosine_similarity": 0.202073
        },
        {
          "label": "load file",
          "cosine_similarity": 0.192545
        },
        {
          "label": "cloud computing",
          "cosine_similarity": 0.186998
        }
      ]
    },
    {
      "id": 57,
      "label": "delete",
      "token_id": 4542,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 9.538492,
        "y2d": 4.107883,
        "x3d": 4.330225,
        "y3d": 0.694164,
        "z3d": 7.942461
      },
      "top5_neighbors": [
        {
          "label": "update",
          "cosine_similarity": 0.305308
        },
        {
          "label": "save",
          "cosine_similarity": 0.293336
        },
        {
          "label": "create",
          "cosine_similarity": 0.288418
        },
        {
          "label": "write",
          "cosine_similarity": 0.236487
        },
        {
          "label": "eat",
          "cosine_similarity": 0.219085
        }
      ]
    },
    {
      "id": 58,
      "label": "eat",
      "token_id": 32066,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 13.93611,
        "y2d": 6.254777,
        "x3d": 5.792569,
        "y3d": -2.180012,
        "z3d": 7.766589
      },
      "top5_neighbors": [
        {
          "label": "watermelon",
          "cosine_similarity": 0.250527
        },
        {
          "label": "encode",
          "cosine_similarity": 0.227456
        },
        {
          "label": "banana",
          "cosine_similarity": 0.220479
        },
        {
          "label": "delete",
          "cosine_similarity": 0.219085
        },
        {
          "label": "grow",
          "cosine_similarity": 0.217416
        }
      ]
    },
    {
      "id": 59,
      "label": "grow",
      "token_id": 66218,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 14.510888,
        "y2d": 7.785308,
        "x3d": 6.628311,
        "y3d": -2.634372,
        "z3d": 6.887133
      },
      "top5_neighbors": [
        {
          "label": "pineapple",
          "cosine_similarity": 0.236973
        },
        {
          "label": "compression",
          "cosine_similarity": 0.221865
        },
        {
          "label": "encryption",
          "cosine_similarity": 0.219186
        },
        {
          "label": "eat",
          "cosine_similarity": 0.217416
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.210604
        }
      ]
    },
    {
      "id": 60,
      "label": "plant",
      "token_id": 20827,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 12.981083,
        "y2d": 5.653416,
        "x3d": 6.930387,
        "y3d": -1.156419,
        "z3d": 8.838923
      },
      "top5_neighbors": [
        {
          "label": "watermelon",
          "cosine_similarity": 0.245724
        },
        {
          "label": "computer vision",
          "cosine_similarity": 0.233208
        },
        {
          "label": "computer",
          "cosine_similarity": 0.222383
        },
        {
          "label": "banana",
          "cosine_similarity": 0.214003
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.212869
        }
      ]
    },
    {
      "id": 61,
      "label": "harvest",
      "token_id": 12982,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 14.644623,
        "y2d": 6.331554,
        "x3d": 5.058346,
        "y3d": -2.908261,
        "z3d": 7.897254
      },
      "top5_neighbors": [
        {
          "label": "pear",
          "cosine_similarity": 0.219306
        },
        {
          "label": "watermelon",
          "cosine_similarity": 0.215821
        },
        {
          "label": "pineapple",
          "cosine_similarity": 0.183952
        },
        {
          "label": "hardware",
          "cosine_similarity": 0.178695
        },
        {
          "label": "eat",
          "cosine_similarity": 0.174993
        }
      ]
    },
    {
      "id": 62,
      "label": "create",
      "token_id": 3182,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 8.778721,
        "y2d": 4.589233,
        "x3d": 4.089892,
        "y3d": 0.836279,
        "z3d": 6.995751
      },
      "top5_neighbors": [
        {
          "label": "delete",
          "cosine_similarity": 0.288418
        },
        {
          "label": "write",
          "cosine_similarity": 0.269559
        },
        {
          "label": "update",
          "cosine_similarity": 0.268708
        },
        {
          "label": "open",
          "cosine_similarity": 0.201594
        },
        {
          "label": "generate text",
          "cosine_similarity": 0.198193
        }
      ]
    },
    {
      "id": 63,
      "label": "update",
      "token_id": 2386,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 9.286003,
        "y2d": 3.958906,
        "x3d": 3.925596,
        "y3d": 0.922254,
        "z3d": 7.706306
      },
      "top5_neighbors": [
        {
          "label": "delete",
          "cosine_similarity": 0.305308
        },
        {
          "label": "create",
          "cosine_similarity": 0.268708
        },
        {
          "label": "write",
          "cosine_similarity": 0.198097
        },
        {
          "label": "sync",
          "cosine_similarity": 0.188009
        },
        {
          "label": "encode",
          "cosine_similarity": 0.175621
        }
      ]
    },
    {
      "id": 64,
      "label": "search",
      "token_id": 1836,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 9.106049,
        "y2d": 5.578289,
        "x3d": 4.590456,
        "y3d": 1.379057,
        "z3d": 6.887807
      },
      "top5_neighbors": [
        {
          "label": "create",
          "cosine_similarity": 0.180971
        },
        {
          "label": "scan document",
          "cosine_similarity": 0.178471
        },
        {
          "label": "store",
          "cosine_similarity": 0.175278
        },
        {
          "label": "render scene",
          "cosine_similarity": 0.173781
        },
        {
          "label": "server",
          "cosine_similarity": 0.155526
        }
      ]
    },
    {
      "id": 65,
      "label": "sync",
      "token_id": 12996,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 10.79126,
        "y2d": 4.007551,
        "x3d": 5.402812,
        "y3d": -0.001326,
        "z3d": 9.125216
      },
      "top5_neighbors": [
        {
          "label": "render scene",
          "cosine_similarity": 0.202008
        },
        {
          "label": "save",
          "cosine_similarity": 0.197208
        },
        {
          "label": "update",
          "cosine_similarity": 0.188009
        },
        {
          "label": "clean cache",
          "cosine_similarity": 0.177919
        },
        {
          "label": "grow",
          "cosine_similarity": 0.175523
        }
      ]
    },
    {
      "id": 66,
      "label": "train model",
      "token_id": 10397,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 12.390308,
        "y2d": 5.932489,
        "x3d": 7.548101,
        "y3d": -0.356747,
        "z3d": 8.852159
      },
      "top5_neighbors": [
        {
          "label": "machine learning",
          "cosine_similarity": 0.226073
        },
        {
          "label": "plant",
          "cosine_similarity": 0.196124
        },
        {
          "label": "render scene",
          "cosine_similarity": 0.157611
        },
        {
          "label": "store",
          "cosine_similarity": 0.151297
        },
        {
          "label": "algorithm",
          "cosine_similarity": 0.148508
        }
      ]
    },
    {
      "id": 67,
      "label": "generate text",
      "token_id": 19366,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 7.319577,
        "y2d": 6.878026,
        "x3d": 6.914622,
        "y3d": 0.274913,
        "z3d": 5.040989
      },
      "top5_neighbors": [
        {
          "label": "text",
          "cosine_similarity": 0.613269
        },
        {
          "label": "analyze data",
          "cosine_similarity": 0.198343
        },
        {
          "label": "create",
          "cosine_similarity": 0.198193
        },
        {
          "label": "format",
          "cosine_similarity": 0.188024
        },
        {
          "label": "scan document",
          "cosine_similarity": 0.184092
        }
      ]
    },
    {
      "id": 68,
      "label": "load file",
      "token_id": 1078,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 8.60006,
        "y2d": 6.481923,
        "x3d": 5.684736,
        "y3d": 1.211631,
        "z3d": 5.979562
      },
      "top5_neighbors": [
        {
          "label": "file",
          "cosine_similarity": 0.608092
        },
        {
          "label": "scan document",
          "cosine_similarity": 0.224896
        },
        {
          "label": "save",
          "cosine_similarity": 0.196573
        },
        {
          "label": "read",
          "cosine_similarity": 0.193642
        },
        {
          "label": "store",
          "cosine_similarity": 0.192545
        }
      ]
    },
    {
      "id": 69,
      "label": "export json",
      "token_id": 1533,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 11.316924,
        "y2d": 4.164443,
        "x3d": 5.622918,
        "y3d": 0.185136,
        "z3d": 9.348258
      },
      "top5_neighbors": [
        {
          "label": "render scene",
          "cosine_similarity": 0.220164
        },
        {
          "label": "save",
          "cosine_similarity": 0.176109
        },
        {
          "label": "sync",
          "cosine_similarity": 0.16988
        },
        {
          "label": "document",
          "cosine_similarity": 0.164974
        },
        {
          "label": "pdf",
          "cosine_similarity": 0.164826
        }
      ]
    },
    {
      "id": 70,
      "label": "analyze data",
      "token_id": 93221,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 8.145755,
        "y2d": 8.208556,
        "x3d": 7.566945,
        "y3d": 0.998637,
        "z3d": 5.490249
      },
      "top5_neighbors": [
        {
          "label": "data",
          "cosine_similarity": 0.580871
        },
        {
          "label": "data science",
          "cosine_similarity": 0.50435
        },
        {
          "label": "database",
          "cosine_similarity": 0.255288
        },
        {
          "label": "encryption",
          "cosine_similarity": 0.240881
        },
        {
          "label": "natural language",
          "cosine_similarity": 0.216433
        }
      ]
    },
    {
      "id": 71,
      "label": "scan document",
      "token_id": 16405,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 9.34067,
        "y2d": 6.71854,
        "x3d": 5.396957,
        "y3d": 1.720407,
        "z3d": 6.387715
      },
      "top5_neighbors": [
        {
          "label": "document",
          "cosine_similarity": 0.645582
        },
        {
          "label": "database",
          "cosine_similarity": 0.248883
        },
        {
          "label": "load file",
          "cosine_similarity": 0.224896
        },
        {
          "label": "pdf",
          "cosine_similarity": 0.209714
        },
        {
          "label": "encode",
          "cosine_similarity": 0.202841
        }
      ]
    },
    {
      "id": 72,
      "label": "render scene",
      "token_id": 7322,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 11.795283,
        "y2d": 4.494968,
        "x3d": 6.175433,
        "y3d": -0.323846,
        "z3d": 9.205539
      },
      "top5_neighbors": [
        {
          "label": "computer vision",
          "cosine_similarity": 0.245118
        },
        {
          "label": "memory",
          "cosine_similarity": 0.229525
        },
        {
          "label": "export json",
          "cosine_similarity": 0.220164
        },
        {
          "label": "save",
          "cosine_similarity": 0.213007
        },
        {
          "label": "compression",
          "cosine_similarity": 0.211732
        }
      ]
    },
    {
      "id": 73,
      "label": "clean cache",
      "token_id": 18377,
      "cluster": "Action/Verb",
      "color": "#D85A30",
      "coords": {
        "x2d": 10.637684,
        "y2d": 4.731027,
        "x3d": 5.454665,
        "y3d": 0.19289,
        "z3d": 8.10795
      },
      "top5_neighbors": [
        {
          "label": "cloud computing",
          "cosine_similarity": 0.231408
        },
        {
          "label": "save",
          "cosine_similarity": 0.214529
        },
        {
          "label": "delete",
          "cosine_similarity": 0.206833
        },
        {
          "label": "render scene",
          "cosine_similarity": 0.195561
        },
        {
          "label": "plant",
          "cosine_similarity": 0.18501
        }
      ]
    }
  ]
};
