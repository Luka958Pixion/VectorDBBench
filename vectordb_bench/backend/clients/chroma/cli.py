from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from .config import ChromaConfig, ChromaHNSWConfig


from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class ChromaTypedDict(TypedDict):
    host: Annotated[
        str, click.option("--host", type=str, help="Db host", required=True)
    ]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]
    port: Annotated[int, click.option("--port", type=int, default=6379, help="Db Port")]


class ChromaHNSWTypedDict(CommonTypedDict, ChromaTypedDict, HNSWFlavor1):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(ChromaHNSWTypedDict)
def Chroma(**parameters: Unpack[ChromaHNSWTypedDict]):
    run(
        db=DB.Chroma,
        db_config=ChromaConfig(
            db_label=parameters["db_label"],
            password=SecretStr(parameters["password"])
            if parameters["password"]
            else None,
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
        ),        
        db_case_config=ChromaHNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_search"],
        ),
        **parameters,
    )
