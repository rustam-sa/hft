"""Add new models

Revision ID: 0c2751aa67e8
Revises: f3ca02425119
Create Date: 2024-06-08 13:22:18.796094

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0c2751aa67e8'
down_revision: Union[str, None] = 'f3ca02425119'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('collections',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('collection_name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('data_frame_metadata',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('label', sa.Integer(), nullable=False),
    sa.Column('collection_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('data_frame_entries',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('data_frame_metadata_id', sa.Integer(), nullable=True),
    sa.Column('open_btc', sa.Float(), nullable=True),
    sa.Column('close_btc', sa.Float(), nullable=True),
    sa.Column('high_btc', sa.Float(), nullable=True),
    sa.Column('low_btc', sa.Float(), nullable=True),
    sa.Column('volume_btc', sa.Float(), nullable=True),
    sa.Column('amount_btc', sa.Float(), nullable=True),
    sa.Column('open_eth', sa.Float(), nullable=True),
    sa.Column('close_eth', sa.Float(), nullable=True),
    sa.Column('high_eth', sa.Float(), nullable=True),
    sa.Column('low_eth', sa.Float(), nullable=True),
    sa.Column('volume_eth', sa.Float(), nullable=True),
    sa.Column('amount_eth', sa.Float(), nullable=True),
    sa.Column('open_btc_robust', sa.Float(), nullable=True),
    sa.Column('close_btc_robust', sa.Float(), nullable=True),
    sa.Column('high_btc_robust', sa.Float(), nullable=True),
    sa.Column('low_btc_robust', sa.Float(), nullable=True),
    sa.Column('volume_btc_robust', sa.Float(), nullable=True),
    sa.Column('amount_btc_robust', sa.Float(), nullable=True),
    sa.Column('open_eth_robust', sa.Float(), nullable=True),
    sa.Column('close_eth_robust', sa.Float(), nullable=True),
    sa.Column('high_eth_robust', sa.Float(), nullable=True),
    sa.Column('low_eth_robust', sa.Float(), nullable=True),
    sa.Column('volume_eth_robust', sa.Float(), nullable=True),
    sa.Column('amount_eth_robust', sa.Float(), nullable=True),
    sa.Column('open_btc_standard', sa.Float(), nullable=True),
    sa.Column('close_btc_standard', sa.Float(), nullable=True),
    sa.Column('high_btc_standard', sa.Float(), nullable=True),
    sa.Column('low_btc_standard', sa.Float(), nullable=True),
    sa.Column('volume_btc_standard', sa.Float(), nullable=True),
    sa.Column('amount_btc_standard', sa.Float(), nullable=True),
    sa.Column('open_eth_standard', sa.Float(), nullable=True),
    sa.Column('close_eth_standard', sa.Float(), nullable=True),
    sa.Column('high_eth_standard', sa.Float(), nullable=True),
    sa.Column('low_eth_standard', sa.Float(), nullable=True),
    sa.Column('volume_eth_standard', sa.Float(), nullable=True),
    sa.Column('amount_eth_standard', sa.Float(), nullable=True),
    sa.Column('open_btc_minmax', sa.Float(), nullable=True),
    sa.Column('close_btc_minmax', sa.Float(), nullable=True),
    sa.Column('high_btc_minmax', sa.Float(), nullable=True),
    sa.Column('low_btc_minmax', sa.Float(), nullable=True),
    sa.Column('volume_btc_minmax', sa.Float(), nullable=True),
    sa.Column('amount_btc_minmax', sa.Float(), nullable=True),
    sa.Column('open_eth_minmax', sa.Float(), nullable=True),
    sa.Column('close_eth_minmax', sa.Float(), nullable=True),
    sa.Column('high_eth_minmax', sa.Float(), nullable=True),
    sa.Column('low_eth_minmax', sa.Float(), nullable=True),
    sa.Column('volume_eth_minmax', sa.Float(), nullable=True),
    sa.Column('amount_eth_minmax', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['data_frame_metadata_id'], ['data_frame_metadata.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('data_frame_entries')
    op.drop_table('data_frame_metadata')
    op.drop_table('collections')
    # ### end Alembic commands ###
