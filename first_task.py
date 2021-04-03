import streamlit as st

st.title("account page")

menu = ["Home", "Login", "amount"]
choice = st.sidebar.selectbox("Menu", menu)
if choice =="Home":
 		st.subheader("Home")


elif choice == "Login":
	st.subheader("Login section")

	account_number = st.sidebar.text_input("account number", )
	security_pin = st.sidebar.text_input("security pin", type= 'password')
	if st.sidebar.checkbox("Login"):
		if account_number == "081" and security_pin == "0000":
			st.success("Logged in as {}".format(account_number) + " please follow the procedure")
			task =  ["deposit", "withdraw", "check_balance"]
			transcation =st.selectbox("Transcation", task)
			st.subheader("enter your details")
			if transcation == "deposit":
				deposit = st.text_input("Enter total deposit")
			elif transcation == "withdraw":
				withdraw = st.text_input("total amount withdraw")
			elif transcation == "check_balance":
				st.subheader("your transcation is been process")
				if st.button("check balance"):
					if deposit > withdraw:
						check_balance = deposit - withdraw
						st.subheader(check_balance)


				#elif transcation == "check_balance":
				#	deposit = "1000"
				#	withdraw = "500"
				#	if deposit >= withdraw:
				#		check_balance = deposit - withdraw
					#	st.subheader(check_balance)

					else: 
						st.subheader("your account is too low to perform this transcation")
		else:
			st.subheader("invild account details")
elif choice == "amount":
	st.subheader("your account balance is 00:00 ")


