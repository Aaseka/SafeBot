import asyncio
import discord
from discord.ext import commands
from discord.ui import Button, View
from transformers import pipeline
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Token
TOKEN = "This is where your token would be stored"

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

# model
classifier = pipeline("text-classification", model="unitary/toxic-bert")

# storing warning counts
warning_counts = {}
MAX_WARNINGS = 3


# muted role
async def mute_user(user, guild):
    muted_role = discord.utils.get(guild.roles, name="Muted")

    if muted_role is None:
        muted_role = await guild.create_role(
            name="Muted", permissions=discord.Permissions(send_messages=False)
        )

    # Set the "Muted" role's permissions in every channel
    for channel in guild.channels:
        await channel.set_permissions(muted_role, send_messages=False)

    # Add the "Muted" role to the user
    await user.add_roles(muted_role)
    await user.send("You have been muted for violating server rules.")


# button clicks
class ActionButtons(View):
    def __init__(self, user, guild, user_id, owner):
        super().__init__(timeout=60)
        self.user = user
        self.guild = guild
        self.user_id = user_id
        self.owner = owner

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """This method ensures only the owner can interact with the buttons."""
        if interaction.user != self.owner:
            await interaction.response.send_message(
                "You are not authorized to use these buttons.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Mute", style=discord.ButtonStyle.primary)
    async def mute_button_callback(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        await mute_user(self.user, self.guild)
        await interaction.response.send_message(
            f"User {self.user.mention} has been muted.", ephemeral=True
        )
        warning_counts[self.user_id] = 0
        self.stop()

    @discord.ui.button(label="Ban", style=discord.ButtonStyle.danger)
    async def ban_button_callback(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        await self.guild.ban(
            self.user, reason="Reached maximum warnings for inappropriate behavior."
        )
        await interaction.response.send_message(
            f"User {self.user.mention} has been banned.", ephemeral=True
        )
        warning_counts[self.user_id] = 0
        self.stop()

    @discord.ui.button(label="Reset Warnings", style=discord.ButtonStyle.secondary)
    async def reset_button_callback(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        warning_counts[self.user_id] = 0
        await interaction.response.send_message(
            f"Warning count reset for {self.user.mention}.", ephemeral=True
        )
        self.stop()


# finding messages
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # going through model
    result = classifier(message.content)
    print(result[0]["score"])

    # checking score (80% thereshold)
    if result[0]["label"] == "toxic" and result[0]["score"] > 0.8:
        user_id = message.author.id

        if user_id in warning_counts:
            warning_counts[user_id] += 1
        else:
            warning_counts[user_id] = 1

        # warning
        await message.channel.send(
            f"{message.author.mention}, this message has been flagged as inappropriate. Warning {warning_counts[user_id]}/{MAX_WARNINGS}"
        )

        await message.delete()

        # action taken
        if warning_counts[user_id] >= MAX_WARNINGS:
            guild_owner = message.guild.owner
            embed = discord.Embed(
                title="User Action Required",
                description=f"User {message.author.mention} has reached the maximum warnings. Please choose an action:",
                color=discord.Color.red(),
            )
            view = ActionButtons(
                user=message.author,
                guild=message.guild,
                user_id=user_id,
                owner=guild_owner,
            )

            # sending embeds to guild
            await message.channel.send(
                content=f"{guild_owner.mention}", embed=embed, view=view
            )


# running command
bot.run(TOKEN)
