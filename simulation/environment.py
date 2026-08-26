from __future__ import annotations
import random

from .agents import FreeAgent
from .referee import RefereeAgent


# ==============  Environment ==============
class Environment:

    def __init__(self):
        self.agents = []
        self.referee = None
        self.round_number = 0
        self.action_log = []

    def add_free_agents(self, agent_configs: list[tuple]):

        for cfg in agent_configs:
            if len(cfg) == 3:
                name, mdl, rst = cfg
            else:
                name, mdl = cfg
                rst = True
            self.agents.append(FreeAgent(name=name, model=mdl, reset_log=rst))

    def add_referee(self, referee_name: str, model: str):
        self.referee = RefereeAgent(name=referee_name, model=model)

    def run_round(self):
        self.round_number += 1
        if not (self.agents and self.referee):
            print("Error: Agents and referee must exist.")
            return

        active_order = self.agents.copy()
        random.shuffle(active_order)
        print(f"\n===== Round {self.round_number} order: {[ag.name for ag in active_order]} =====")

        for active in active_order:
            others = [ag.name for ag in self.agents if ag != active]
            context_info = ", ".join(others)

            main_text   = active.decide_next(self.round_number, context_info)
            action_data = self.referee.parse_text(active.name, main_text)

            self.action_log.append({
                "round":          self.round_number,
                "type":           "main",
                "actor":          active.name,
                "raw_text":       main_text,
                "action_name":    action_data["action_name"],
                "is_interaction": action_data["is_interaction"],
                "valence":        action_data["valence"],
                "description":    action_data["description"],
                "group_invitation": action_data.get("group_invitation", False)
            })

            if not action_data["is_interaction"]:
                continue


            if not action_data["group_invitation"]:
                targets = action_data.get("targets") or [ag.name for ag in self.agents if ag != active]
                broadcast_text = main_text  # A's original sentence

                for other in (ag for ag in self.agents if ag.name in targets):
                    other.receive_message(
                        self.round_number,
                        from_agent=active.name,
                        content=broadcast_text
                    )

                    reaction_text = other.decide_reaction(
                        self.round_number,
                        actor_name=active.name,
                        action_name=action_data["action_name"],
                        referee_desc=action_data.get("description", "") or "Responds appropriately.",
                        display_text=main_text,  
                    )

                    active.receive_message(
                        self.round_number,
                        from_agent=other.name,
                        content=reaction_text
                    )

                    reaction_data = self.referee.parse_text(other.name, reaction_text)
                    self.action_log.append({
                        "round":          self.round_number,
                        "type":           "reaction",
                        "actor":          other.name,
                        "raw_text":       reaction_text,
                        "action_name":    reaction_data["action_name"],
                        "is_interaction": reaction_data["is_interaction"],
                        "valence":        reaction_data["valence"],
                        "description":    reaction_data["description"]
                    })
                # --------------------------------
            else:

                if len(self.agents) != 3:
                    targets = action_data.get("targets") or [
                        ag.name for ag in self.agents if ag != active
                    ]

                    for other in (ag for ag in self.agents if ag.name in targets):
                        other.receive_message(
                            self.round_number,
                            from_agent=active.name,
                            content=main_text,
                        )

                        reaction_text = other.decide_reaction(
                            self.round_number,
                            actor_name=active.name,
                            action_name=action_data["action_name"],
                            referee_desc=action_data.get("description", "") or "Responds appropriately.",
                            display_text=main_text, 
                        )
                        reaction_data = self.referee.parse_text(other.name, reaction_text)
                        reaction_log = {
                            "round":          self.round_number,
                            "type":           "reaction",
                            "actor":          other.name,
                            "raw_text":       reaction_text,
                            "action_name":    reaction_data["action_name"],
                            "is_interaction": reaction_data["is_interaction"],
                            "valence":        reaction_data["valence"],
                            "description":    reaction_data["description"],
                            "agree_to_group": reaction_data.get("agree_to_group", False),
                        }
                        self.action_log.append(reaction_log)

                        active.receive_message(
                            self.round_number,
                            from_agent=other.name,
                            content=reaction_text,
                        )

                else:
                    others = [ag for ag in self.agents if ag != active]
                    random.shuffle(others)
                    first_partner, second_partner = others[0], others[1]
                    partners = [first_partner, second_partner]

                    for listener in partners:
                        listener.receive_message(
                            self.round_number,
                            from_agent=active.name,
                            content=main_text, 
                        )

                    decisions = []  
                    for partner in partners:
                        resp_text = partner.decide_reaction(
                            self.round_number,
                            actor_name=active.name,
                            action_name=action_data["action_name"],
                            referee_desc=action_data.get("description", "") or "Responds appropriately.",
                            display_text=main_text, 
                        )
                        resp_data = self.referee.parse_text(partner.name, resp_text)
                        decisions.append({
                            "agent": partner,
                            "text":  resp_text,
                            "data":  resp_data,
                        })

                        group_log = {
                            "round":          self.round_number,
                            "type":           "group_interaction",
                            "actor":          partner.name,
                            "raw_text":       resp_text,
                            "action_name":    resp_data["action_name"],
                            "is_interaction": resp_data["is_interaction"],
                            "valence":        resp_data["valence"],
                            "description":    resp_data["description"],
                            "agree_to_group": resp_data.get("agree_to_group", False),
                        }
                        self.action_log.append(group_log)

                    agreed_partners = [
                        d["agent"] for d in decisions
                        if d["data"].get("agree_to_group", False)
                    ]

                    if not agreed_partners:
                        for d in decisions:
                            partner = d["agent"]
                            text    = d["text"]
                            active.receive_message(
                                self.round_number,
                                from_agent=partner.name,
                                content=text,
                            )
                    else:
                        for d in decisions:
                            partner = d["agent"]
                            text    = d["text"]
                            data    = d["data"]
                            if not data.get("agree_to_group", False):
                                active.receive_message(
                                    self.round_number,
                                    from_agent=partner.name,
                                    content=text,
                                )
                                continue

                            listeners = [active] + [
                                p for p in agreed_partners if p is not partner
                            ]
                            for listener in listeners:
                                listener.receive_message(
                                    self.round_number,
                                    from_agent=partner.name,
                                    content=text,
                                )




    def print_log(self):
        for entry in self.action_log:
            r = entry.get("round", "N/A")
            etype = entry.get("type", "N/A")
            actor = entry.get("actor", "N/A")
            raw = entry.get("raw_text", "N/A")
            aname = entry.get("action_name", "N/A")
            inter = entry.get("is_interaction", "N/A")
            val = entry.get("valence", "N/A")
            desc = entry.get("description", "N/A")
            g_inv = entry.get("group_invitation", None)
            agree = entry.get("agree_to_group", None)
            print(f"[Round {r}] ({etype.upper()}) {actor} said: '{raw}'")
            print(f"    -> action_name={aname}, is_interaction={inter}, valence={val}, desc={desc}")
            if g_inv is not None:
                print(f"    -> group_invitation={g_inv}")
            if agree is not None:
                print(f"    -> agree_to_group={agree}")

    # def show_agent_logs(self):
    #     for ag in self.agents:
    #         print(f"\n=== Logs for {ag.name} ({ag.log_filename}) ===")
    #         if not os.path.exists(ag.log_filename):
    #             continue
    #         with open(ag.log_filename, "r", encoding="utf-8") as f:
    #             content = f.read()
    #         print(content)
