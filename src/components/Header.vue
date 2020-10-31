<template>
  <div class="header">
    <h1 class="header__title">Menu</h1>
    <ul class="header__list">
      <li class="header__list-item"></li>
      <li class="header__list-item"></li>
      <li class="header__list-item"></li>
      <li class="header__list-item"></li>
      <li class="header__list-item"></li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'Header',
  // props: {
  //   msg: String
  // }
}


</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
div.header {
  margin: 0 auto;
  width: 90%;
  background-color: rgba(0, 98, 255, .6);
  display: flex;
  justify-content: flex-start;
  border-radius: 5px;
  padding: 0 10px;
} 
h1.header__title {
text-align: center;
}
ul.header__list {
  /* display: none; */
  list-style-type: none;
  padding: 0;
}
li.header__list-item {
  display: inline-block;
  margin: 0 10px;
}
/* a {
  color: #42b983;
} */
</style>
